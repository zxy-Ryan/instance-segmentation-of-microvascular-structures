import json, cv2, numpy as np, itertools, random, pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn import model_selection

import matplotlib.pyplot as plt
from skimage import io
from pycocotools.coco import COCO
import matplotlib.patches as mpatches

def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_to_binary_mask(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

## Loading Datase

df = pd.read_csv('../datasets/hubmap2023/tile_meta.csv')
df = df.query('dataset != 3')
#df=df.head(500)
df.reset_index(inplace=True,drop=True)
# print(df.head())

## Spliting training & Valid
from sklearn.model_selection import StratifiedKFold

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['source_wsi']), 1):
    df.loc[val_idx, 'fold'] = fold

df['fold'] = df['fold'].astype(np.uint8)
# print(df.groupby('fold').size())

selected_fold = 5
train_ids = df.query(f'fold != {selected_fold}')['id'].values.tolist()
valid_ids = df.query(f'fold == {selected_fold}')['id'].values.tolist()
# print(len(train_ids), len(valid_ids))

## Reading polygons.jsonl
jsonl_file_path = "../datasets/hubmap2023/polygons.jsonl"
data = []
with open(jsonl_file_path, "r") as file:
    for line in file:
        data.append(json.loads(line))

## Cateogories
# categories_list=['blood_vessel','glomerulus'] ## 2 class
# categories_list=['blood_vessel'] ## 1 class
categories_list=['blood_vessel','glomerulus', 'unsure'] ## 3 class
#------------------------------------------------------------------------------
categories_ids = {name:id+1 for id, name in enumerate(categories_list)}
ids_categories = {id+1:name for id, name in enumerate(categories_list)}
categories =[{'id':id,'name':name} for name,id in categories_ids.items()]

# print(categories_ids)
# print(ids_categories)
# print(categories)

## Creating COCO
def coco_structure(images_ids):
    idx = 1
    annotations = []
    images = []
    for item in tqdm(data, total=int(len(images_ids))):
        image_id = item["id"]
        if image_id in images_ids:
            image = {"id": image_id, "file_name": image_id + ".tif", "height": 512, "width": 512}
            images.append(image)
        else:
            continue
        # -----------------------------
        anns = item["annotations"]
        for an in anns:
            category_type = an["type"]
            # if category_type != "unsure":
            if category_type in categories_list:
                category_id = categories_ids[category_type]
                segmentation = an["coordinates"]
                mask_img = coordinates_to_masks(segmentation, (512, 512))[0]
                ys, xs = np.where(mask_img)
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                rle = binary_mask_to_rle(mask_img)

                seg = {
                    "id": idx,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                    "area": int(np.sum(mask_img)),
                    "iscrowd": 0,
                }
                if image_id in images_ids:
                    annotations.append(seg)
                    idx = idx + 1

    return {"info": {}, "licenses": [], "categories": categories, "images": images, "annotations": annotations}

train_coco_data = coco_structure(train_ids)
valid_coco_data = coco_structure(valid_ids)


## Saving COCO
output_file_path = f"../datasets/hubmap2023/annotations_json_c3/coco_annotations_train_all_fold{selected_fold}.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(train_coco_data, output_file, ensure_ascii=True, indent=4)

output_file_path = f"../datasets/hubmap2023/annotations_json_c3/coco_annotations_valid_all_fold{selected_fold}.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(valid_coco_data, output_file, ensure_ascii=True, indent=4)

## Visualization
dataDir = Path("../datasets/hubmap2023/train")
annFile = Path(f"../datasets/hubmap2023/annotations_json_c3/coco_annotations_valid_all_fold{selected_fold}.json")

colors = ['Set1','Set3_r','Set3']
legend = ids_categories #{1: 'blood_vessel',2:glomerulus}

coco = COCO(annFile)
imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds[0:4])

fig, axs = plt.subplots(len(imgs), 2, figsize=(10, 5*len(imgs)))
for img, ax_row in zip(imgs, axs):
    ax = ax_row[0]  # Access the first axis in each row
    I = io.imread(dataDir / img["file_name"])
    annIds = coco.getAnnIds(imgIds=[img["id"]])
    anns = coco.loadAnns(annIds)
    ax.imshow(I)
    ax = ax_row[1]  # Access the second axis in each row
    ax.imshow(I)
    plt.sca(ax)
    for i, ann in enumerate(anns):
        category_id = ann['category_id']
        color = colors[category_id-1]
        #-----------------------------------------
        mask = coco.annToMask(ann)
        mask = np.ma.masked_where(mask == 0, mask)
        ax.imshow(mask, cmap=color, alpha=0.8)
        #-----------------------------------------
        handles = []
        for category_id in legend:
            color = colors[category_id - 1]
            handles.append(mpatches.Patch(color=plt.colormaps.get_cmap(color)(0)))
        ax.legend(handles, legend.values(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.axis('off')
plt.show()
# fig.savefig('./outfig/figure.png')