import logging
import os
import json
import cv2
import random
import copy
import torch
from collections import OrderedDict
from tqdm import tqdm

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.utils.comm import is_main_process
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format
)

logger = logging.getLogger("detectron2")

################################################################################
# Registring the datasets
register_coco_instances("coco_train", {}, "./datasets/COCO/annotations/instances_train2017_edited.json", "./datasets/COCO/train2017")
register_coco_instances("coco_test", {}, "./datasets/COCO/annotations/instances_val2017_edited.json", "./datasets/COCO/val2017")

MetadataCatalog.get("coco_train").set(thing_classes=["apple"])
MetadataCatalog.get("coco_test").set(thing_classes=["apple"])

# Testing if the datasets are registered properly
coco_metadata = MetadataCatalog.get("coco_train")
dataset_dicts = get_detection_dataset_dicts(["coco_train"])
i = 1
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=coco_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("output" + str(i) + ".jpg",vis.get_image()[:, :, ::-1])
    i += 1

################################################################################
def do_train(cfg, model, resume=False):
    """
    MAIN TRAINING LOOP
    """

    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # data_loader = build_detection_train_loader(cfg, mapper=mapper)
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

################################################################################
def do_test(cfg, model):
    """
    MAIN TESTING LOOP
    """

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 # set the testing threshold for this model

    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        # print(dataset_name)
        # data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, cfg, True, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]

    return results

################################################################################
def setup(args):
    """
    CREATE CONFIGS AND PERFORM BASIC SETUPS
    """

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ("coco_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00010
    cfg.SOLVER.MAX_ITER = 6000    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 100   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

################################################################################
def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(cfg.OUTPUT_DIR, "model_final.pth"), resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model)
    return do_test(cfg, model)

if __name__=="__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
