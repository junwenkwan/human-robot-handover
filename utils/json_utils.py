import json
import os
import argparse
import natsort
import pandas as pd

def process_dicts(json_dir):
    files = os.listdir(json_dir)
    files = natsort.natsorted(files)
    print(files)
    dict_arr = []
    list = ['0_','1_','2_','3_','5_']

    for file in files:
        json_file = os.path.join(json_dir, file)
        print(json_file)

        with open(json_file) as f:
            json_file = json.load(f)
            annos = json_file['annotation']

            if os.path.splitext(file)[0][6:8] in list:
                # get every 20th frame
                for index, element in enumerate(annos):
                    if index%20 == 0:
                        element['video_name'] = file
                        dict_arr.append(element)
            else:
                # get every 10th frame
                for index, element in enumerate(annos):
                    if index%10 == 0:
                        element['video_name'] = file
                        dict_arr.append(element)

    return dict_arr

def process_csv(csv_path):
    filename_arr = []
    label_arr = []
    csv = pd.read_csv(csv_path)
    csv_dict = csv.to_dict('records')
    for index,element in enumerate(csv_dict):
        filename = element['file_name']
        label = element['label']
        filename_arr.append(filename)
        label_arr.append(label)

    return label_arr


def main(args):
    dict = process_dicts(args.json_path[0])
    label = process_csv(args.csv_path[0])

    for index,element in enumerate(dict):
        element['label'] = label[index]
  
    with open(args.output_json[0], 'w') as output_file:
        json.dump(dict, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train feature vector network"
    )
    parser.add_argument(
        "--json-path",
        default="./",
        nargs=1,
        metavar="JSON_PATH",
        help="Path to the json file",
        type=str
    )
    parser.add_argument(
        "--csv-path",
        default="./",
        nargs=1,
        metavar="CSV_PATH",
        help="Path to the json file",
        type=str
    )
    parser.add_argument(
        "--output-json",
        default="./",
        nargs=1,
        metavar="OUTPUT_JSON",
        help="Path to the output json file",
        type=str
    )
    args = parser.parse_args()

    main(args)
