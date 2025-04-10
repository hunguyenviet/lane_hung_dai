import os

def replace_in_file(file_path, old_str1, new_str1, old_str2='###', new_str2='###'):
  with open(file_path, 'r') as file:
      config_content = file.readlines()

  new_cut_height = new_str1
  new_sample_y = new_str2

  # Replace old_str -> new_str
  for i, line in enumerate(config_content):
      if old_str1 in line:
          config_content[i] = new_cut_height
      if old_str2 in line: ##chưa đổi
          config_content[i] = new_sample_y
  # Rewrite new text in file
  with open(file_path, 'w') as file:
      file.writelines(config_content)

#Replace in resnet18_culane.py
config_path = '/content/lanedet/configs/ufld/resnet18_culane.py'
# replace_in_file(config_path, 'cut_height=0', 'cut_height=ori_img_h//4\n', 'sample_y = range(589, 230, -20)','sample_y = range(ori_img_h-1, cut_height, -15)\n')
replace_in_file(config_path, "dataset_path = './data/CULane'", "dataset_path = '/content/dataset'\n")
replace_in_file(config_path, "epochs = 50", "epochs = 650\n")

#Replace in mobilenet.py
file_path = '/content/lanedet/lanedet/models/backbones/mobilenet.py'
replace_in_file(file_path, 'from torchvision.models.utils import load_state_dict_from_url', 'from torch.hub import load_state_dict_from_url\n')

#Replace in transforms.py
file_path = '/content/lanedet/lanedet/datasets/process/transforms.py'
replace_in_file(file_path, 'assert (isinstance(size, collections.Iterable) and len(size) == 2)', '        assert (isinstance(size, collections.abc.Iterable) and len(size) == 2)\n') #thiếu chuyên nghiệp....