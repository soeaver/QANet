from utils.data import transforms as T

from instance.core.config import cfg


def build_transforms(is_train=True):
    # box 2 cs
    image_width, image_height = cfg.TRAIN.SCALES[0]
    aspect_ratio = image_width * 1.0 / image_height
    keypoint_pixel_std = cfg.KEYPOINT.PIXEL_STD

    # heatmap
    target_type = cfg.KEYPOINT.TARGET_TYPE
    sigma = cfg.KEYPOINT.SIGMA
    heatmap_size = cfg.KEYPOINT.HEATMAP_SIZE
    train_size = cfg.TRAIN.SCALES[0]
    test_size = cfg.TEST.SCALE

    # normalize
    pixel_mean = cfg.PIXEL_MEANS
    pixel_std = cfg.PIXEL_STDS
    
    if is_train:
        # scale, rotate, flip
        scale_factor = cfg.TRAIN.SCALE_FACTOR
        rotation_factor = cfg.TRAIN.ROT_FACTOR
        flip = cfg.TRAIN.USE_FLIPPED

        # MSPN halfbody
        half_body = cfg.TRAIN.USE_HALF_BODY
        num_keypoints_half_body = cfg.TRAIN.NUM_KEYPOINTS_HALF_BODY
        prob_half_body = cfg.TRAIN.PRO_HALF_BODY
        points_num = cfg.KEYPOINT.NUM_JOINTS
        upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        x_ext_half_body = cfg.TRAIN.X_EXT_HALF_BODY
        y_ext_half_body = cfg.TRAIN.Y_EXT_HALF_BODY
        
        transform = T.Compose(
            [
                # For the whole image
                T.Box2CS(aspect_ratio, keypoint_pixel_std),
                T.Half_Body(half_body, num_keypoints_half_body, prob_half_body, points_num, upper_body_ids,
                            x_ext_half_body, y_ext_half_body, aspect_ratio, keypoint_pixel_std),
                T.Scale(scale_factor),
                T.Rotate(rotation_factor),
                T.Flip(flip),

                T.Affine(train_size),

                # Normalize
                T.ToTensor(),
                T.BGR_Normalize(pixel_mean, pixel_std, to_rgb=False),

                # Generate target tensor from the class
                T.Generate_Target(target_type, sigma, heatmap_size, train_size),
            ]
        )

    else:
        transform = T.Compose(
            [
                # For the whole image
                T.Box2CS(aspect_ratio, keypoint_pixel_std),

                T.Affine(test_size),

                # Normalize
                T.ToTensor(),
                T.BGR_Normalize(pixel_mean, pixel_std, to_rgb=False),

                # Generate target tensor from the class
                T.Generate_Target(target_type, sigma, heatmap_size, test_size),
            ]
        )
    return transform
