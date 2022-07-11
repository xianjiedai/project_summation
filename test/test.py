from __future__ import print_function
import os
import torch
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
from networks.seg_net import Segnet
from data_preprocessing.data_from_prepared_dataset import input_transform, colorize_mask

#select only img files and filter other files, such as txt or csv, out
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])

#choose model and the path of trained model
model = Segnet(3,2)
model_path = 'checkpoint/Segnet/model/netG_final.pth'
#path of test set
test_path = 'data/test_set_images/'
test_set = os.listdir(test_path)
# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25
#path of submission CSV file
submission_filename = 'submission/submission.csv'
image_filenames = []

'''
# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))
'''

#used trained model to predict test set
for i in tqdm(range(len(test_set))):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    test_image_path = test_path + str(test_set[i]) + '/' +str(test_set[i]) + '.png'
    test_image = Image.open(test_image_path).convert('RGB')
    print('Operating...')
    img = input_transform(test_image)
    img = img.unsqueeze(0)
    img = Variable(img)
    pred_image = model(img)
    predictions = pred_image.data.max(1)[1].squeeze_(1).cpu().numpy()
    prediction = predictions[0]
    predictions_color = colorize_mask(prediction)
    save_name = str(test_set[i])
    print(save_name)
    predictions_color.save("data_segnet/prediction/" + str(save_name) + '.png')

#create submission CSV file
for i in range(1, 51):
    image_filename = 'data_segnet/prediction/test_' + str(i) + '.png'
    print(image_filename)
    image_filenames.append(image_filename)
#masks_to_submission(submission_filename, *image_filenames)


