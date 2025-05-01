import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
"""
Evaluator for interactive single-object segmentation
"""
scene_object_set = set()
class EvaluatorSO():

    def __init__(
        self,
        result_file,
        MAX_IOU):
        
        self.MAX_IOU = MAX_IOU
        self.result_file = result_file
        self.MAX_OBJECT_IOU = defaultdict(int)

    def eval_per_class(self, MAX_IOU=0.8):
        objects = {}

        results_dict_KatIOU = {}
        num_objects = 0
        ordered_clicks = []

        all_object={}
        results_dict_per_click = {}
        results_dict_per_click_iou = {}
        all={}
        with open(self.result_file, 'r') as f:
            while True:
            # for i in range(100):
                line = f.readline()
                if not line:
                    break
                splits = line.rstrip().split(' ')
                scene_name = splits[1].replace('scene','')
                object_id = splits[2]
                num_clicks = splits[3]
                iou=splits[4]

                scene_object_set.add((scene_name, object_id))

                all_object[(scene_name + '_' + object_id)]=1
                all[(scene_name + '_' + object_id)]=[]
                all[(scene_name + '_' + object_id)].append((num_clicks,iou))

                self.MAX_OBJECT_IOU[(scene_name + '_' + object_id)] = max(self.MAX_OBJECT_IOU[(scene_name + '_' + object_id)], float(iou))
                if float(iou)>=MAX_IOU:
                    if (scene_name+'_'+object_id) not in results_dict_KatIOU:
                        results_dict_KatIOU[scene_name+'_'+object_id]=float(num_clicks)
                        num_objects+=1
                        ordered_clicks.append(float(num_clicks))

                # elif int(float(num_clicks))>=20 and (float(iou)>=0):
                #     if (scene_name+'_'+object_id) not in results_dict_KatIOU:
                #         results_dict_KatIOU[scene_name+'_'+object_id] = float(num_clicks)
                #         num_objects += 1
                #         ordered_clicks.append(float(num_clicks))

                results_dict_per_click.setdefault(num_clicks, 0)
                results_dict_per_click_iou.setdefault(num_clicks, 0)

                results_dict_per_click[num_clicks]+=1
                results_dict_per_click_iou[num_clicks]+=float(iou)
                # else:
                #     print(f"Object {scene_name + '_' + object_id} not found in results file")
                #     pass
                # print(f"{results_dict_KatIOU}")
                # print(all_object)
                # print(all)
                # print(results_dict_KatIOU)
        if len(results_dict_KatIOU.values())==0:
            print('no objects to eval')
            return 0
        
        # print(f"{results_dict_KatIOU=}")
        # print(f"{results_dict_per_click=}")
        # print(f"{results_dict_per_click_iou=}")


        click_at_IoU =sum(results_dict_KatIOU.values())/len(results_dict_KatIOU.values())
        print('click@', MAX_IOU, click_at_IoU, num_objects, len(results_dict_KatIOU.values()))


        return ordered_clicks, sum(results_dict_KatIOU.values()), len(results_dict_KatIOU.values()), results_dict_per_click_iou, results_dict_per_click 


    def eval_results(self):
        print('--------- Evaluating -----------')
        NOC = {}
        NOO = {}
      
        for iou_max in self.MAX_IOU:
            NOC[iou_max] = []
            NOO[iou_max] = []
            IOU_PER_CLICK_dict = None
            NOO_PER_CLICK_dict = None

                
            try:
                _, noc_perclass, noo_perclass, iou_per_click, noo_per_click = self.eval_per_class(iou_max)
                NOC[iou_max].append(noc_perclass)
                NOO[iou_max].append(noo_perclass)

                if IOU_PER_CLICK_dict == None:
                    IOU_PER_CLICK_dict = iou_per_click
                else:
                    for k in IOU_PER_CLICK_dict.keys():
                        IOU_PER_CLICK_dict[k] += iou_per_click[k]

                if NOO_PER_CLICK_dict == None:
                    NOO_PER_CLICK_dict = noo_per_click
                else:
                    for k in NOO_PER_CLICK_dict.keys():
                        NOO_PER_CLICK_dict[k] += noo_per_click[k]
            except Exception as e:
                print(f"Error evaluating class {iou_max}: {e}")

        print(f"{NOC=}")
        print(f"{NOO=}")

        NoOs = [NOO[key][0] for key in self.MAX_IOU]
        print(f"{NoOs=}")
        # for i in range(len(NoOs) - 2, -1, -1):
        #     print(i)
        #     print(self.MAX_IOU[i])
        #     NoOs[i] = NoOs[i] + NoOs[i+1]
        print(f"{NoOs=}")
        # print(f"{i=}")
        # print(f"{MAX_OBJECT_IOU=}")



        results_dict = {
            'Avg_IoU': sum(self.MAX_OBJECT_IOU.values())/len(self.MAX_OBJECT_IOU),
            'Reached@0': NoOs[self.MAX_IOU.index(0.001)],
            'Reached@50': NoOs[self.MAX_IOU.index(0.5)],
            'Reached@65': NoOs[self.MAX_IOU.index(0.65)],
            'Reached@80': NoOs[self.MAX_IOU.index(0.8)],
            'Reached@85': NoOs[self.MAX_IOU.index(0.85)],
            'Reached@90': NoOs[self.MAX_IOU.index(0.9)],
            'Reached%50': NoOs[self.MAX_IOU.index(0.5)]/NoOs[self.MAX_IOU.index(0.001)],
            'Reached%65': NoOs[self.MAX_IOU.index(0.65)]/NoOs[self.MAX_IOU.index(0.001)],
            'Reached%80': NoOs[self.MAX_IOU.index(0.8)]/NoOs[self.MAX_IOU.index(0.001)],
            'Reached%85': NoOs[self.MAX_IOU.index(0.85)]/NoOs[self.MAX_IOU.index(0.001)],
            'Reached%90': NoOs[self.MAX_IOU.index(0.9)]/NoOs[self.MAX_IOU.index(0.001)],
            'NoC@50': sum(NOC[0.5])/sum(NOO[0.5]) if sum(NOO[0.5]) != 0 else 0,
            'NoC@65': sum(NOC[0.65])/sum(NOO[0.65]) if sum(NOO[0.65]) != 0 else 0,
            'NoC@80': sum(NOC[0.8])/sum(NOO[0.8]) if sum(NOO[0.8]) != 0 else 0,
            'NoC@85': sum(NOC[0.85])/sum(NOO[0.85]) if sum(NOO[0.85]) != 0 else 0,
            'NoC@90': sum(NOC[0.9])/sum(NOO[0.9]) if sum(NOO[0.9]) != 0 else 0,
            'IoU@1': IOU_PER_CLICK_dict['1']/NOO_PER_CLICK_dict['1'] if IOU_PER_CLICK_dict is not None and NOO_PER_CLICK_dict is not None and '1' in NOO_PER_CLICK_dict and NOO_PER_CLICK_dict['1'] != 0 else 0,
            'IoU@2': IOU_PER_CLICK_dict['2']/NOO_PER_CLICK_dict['2'] if IOU_PER_CLICK_dict is not None and NOO_PER_CLICK_dict is not None and '2' in NOO_PER_CLICK_dict and NOO_PER_CLICK_dict['2'] != 0 else 0,
            'IoU@3': IOU_PER_CLICK_dict['3']/NOO_PER_CLICK_dict['3'] if IOU_PER_CLICK_dict is not None and NOO_PER_CLICK_dict is not None and '3' in NOO_PER_CLICK_dict and NOO_PER_CLICK_dict['3'] != 0 else 0,
            'IoU@5': IOU_PER_CLICK_dict['5']/NOO_PER_CLICK_dict['5'] if IOU_PER_CLICK_dict is not None and NOO_PER_CLICK_dict is not None and '5' in NOO_PER_CLICK_dict and NOO_PER_CLICK_dict['5'] != 0 else 0,
        }
        print('****************************')
        print(results_dict)

        print(self.MAX_OBJECT_IOU.values())

        return results_dict

result_files = [
    "/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/results/ScanNetPP_short_single/val_results_single_SAMPro3D_seg_thresh_0.5.csv",
    "/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/results/ScanNetPP_short_single/val_results_single_SAMPro3D_seg_thresh_0.5_neg_bg.csv",
    "/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/results/ScanNetPP_short_single/val_results_single_SAMPro3D_SAM2_seg_thresh_0.5.csv",
    "/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/results/ScanNetPP_short_single/val_results_single_SAMPro3D_SAM2_seg_thresh_0.5_neg_bg.csv",
    "/home/kuzhum/sam/promptable-3d-object-segmentation/AGILE3D/results/ScanNetPP_short_single/val_results_single_AGILE3D.csv",
]

exp_names = [
    "SAM",
    "SAM + neg. bg.",
    "SAM 2",
    "SAM 2 + neg. bg.",
    "AGILE3D",
]

results_dicts = {}

for result_file, exp_name in zip(result_files, exp_names):
    print(f"Evaluating {exp_name} with {result_file}")
    evaluator = EvaluatorSO(result_file=result_file, MAX_IOU=[0.001,0.5,0.65,0.8,0.85,0.9])
    results_dict = evaluator.eval_results()
    results_dicts[exp_name] = results_dict

print(results_dicts)

print("Avg IoU values:")
# Print out all Avg IoU values
for exp_name, results_dict in results_dicts.items():
    print(f"{exp_name}: {results_dict['Avg_IoU'] * 100 :.2f}")

print("\n\n")

# Print out all Reached@0 values
# for exp_name, results_dict in results_dicts.items():
#     print(f"{exp_name}: {results_dict['Reached@0']}")

# For each exp_name, print the next line
#     SAM & 27.93 & 16.20 & 4.47 & 3.35 & 1.68 & 1.20 & 1.48 & 1.25 & 1.50 & 1.33  \\
# Where first goes 5 values of Reached%* values, delimited by &, then 5 values of NoC@* values, delimited by &

print("Reached%* values:")
for exp_name, result_dict in results_dicts.items():
    print(f"{exp_name} & {result_dict['Reached%50'] * 100 :.1f} & {result_dict['Reached%65'] * 100 :.1f} & {result_dict['Reached%80'] * 100 :.1f} & {result_dict['Reached%85'] * 100 :.1f} & {result_dict['Reached%90'] * 100 :.1f} & {result_dict['NoC@50'] :.2f} & {result_dict['NoC@65'] :.2f} & {result_dict['NoC@80'] :.2f} & {result_dict['NoC@85'] :.2f} & {result_dict['NoC@90'] :.2f} \\\\")

print("\n\n")

# For each exp_name, print the next line
# SAM & 56 & 68 & 72 & 78 \\  % ← fill in
# Where values are IoU@* values, delimited by &

print("IoU@* values:")
for exp_name, result_dict in results_dicts.items():
    print(f"{exp_name} & {result_dict['IoU@1'] * 100 :.2f} & {result_dict['IoU@2'] * 100 :.2f} & {result_dict['IoU@3'] * 100 :.2f} & {result_dict['IoU@5'] * 100 :.2f} \\\\")

print("\n\n")

# print("Reached%* values:")
# # Print out for each exp_name, all Reached%* values, delimited by &
# for exp_name, results_dict in results_dicts.items():
#     print(f"{exp_name}: {results_dict['Reached%50'] * 100 :.2f} & {results_dict['Reached%65'] * 100 :.2f} & {results_dict['Reached%80'] * 100 :.2f} & {results_dict['Reached%85'] * 100 :.2f} & {results_dict['Reached%90'] * 100 :.2f}")


# print("NoC@* values:")
# # Print out for each exp_name, all NoC@* values, delimited by &
# for exp_name, results_dict in results_dicts.items():
#     print(f"{exp_name}: {results_dict['NoC@50'] :.2f} & {results_dict['NoC@65'] :.2f} & {results_dict['NoC@80'] :.2f} & {results_dict['NoC@85'] :.2f} & {results_dict['NoC@90'] :.2f}")

# print("IoU@* values:")
# # Print out for each exp_name, all IoU@* values, delimited by &
# for exp_name, results_dict in results_dicts.items():
#     print(f"{exp_name}: {results_dict['IoU@1'] * 100 :.2f} & {results_dict['IoU@2'] * 100 :.2f} & {results_dict['IoU@3'] * 100 :.2f} & {results_dict['IoU@5'] * 100 :.2f}")

print(f"{scene_object_set=}")