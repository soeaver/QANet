# Data Prepare

This document covers how to prepare training and evaluating data for QANet, including CIHP and LIP dataset.



# Data Structure

  We strongly recommend downloading the data format we prepared.

  - [PASCAL-Person-Part](https://drive.google.com/file/d/1XQsTBz7Rwp38qsIYtU9X0LlrQN84eODX/view?usp=sharing)
  - [CIHP](https://drive.google.com/file/d/1k-cUS2WyK9eAasYcheCTmqSgLSaEO4U4/view?usp=sharing)
  - [LIP](https://drive.google.com/file/d/15buYPS1syjkn1r9NWMA5YfKn4VtpL6hT/view?usp=sharing)
  
  Make sure to put the files as the following structure:

  ```
  ├─cfgs
  ├─ckpts
  ├─data
  │  ├─PASCAL-Person-Part
  │  │  ├─Training
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─Testing
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─annotations
  │  │  │  ├─PASCAL-Person-Part_train.json
  │  │  │  ├─PASCAL-Person-Part_test.json
  │  ├─CIHP
  │  │  ├─Training
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─Validation
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─annotations
  │  │  │  ├─CIHP_train.json
  │  │  │  ├─CIHP_val.json
  │  ├─LIP
  │  │  ├─Training
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─Validation
  │  │  │  ├─Categories
  │  │  │  ├─Category_ids
  │  │  │  ├─Human_ids
  │  │  │  ├─Humans
  │  │  │  ├─Images
  │  │  │  ├─Instance_ids
  │  │  │  ├─Instances
  │  │  ├─annotations
  │  │  │  ├─LIP_train.json
  │  │  │  ├─LIP_val.json
  ├─docs
  ├─instance
  ├─lib
  ├─tools
  ├─weights
     ├─resnet50c-pretrained.pth
     ├─resnet101d-pretrained.pth
     ├─hrnetv2_w48_imagenet_pretrained.pth

  ```
