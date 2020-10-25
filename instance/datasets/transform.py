from lib.data.transforms import transforms as T


def build_transforms(cfg, is_train=True):
    image_format = cfg.IMAGE_FORMAT
    # keypoints prob map
    target_type = cfg.KEYPOINT.TARGET_TYPE
    sigma = cfg.KEYPOINT.SIGMA
    prob_size = cfg.KEYPOINT.PROB_SIZE

    # normalize
    pixel_mean = cfg.PIXEL_MEANS
    pixel_std = cfg.PIXEL_STDS

    if is_train:
        train_size = cfg.TRAIN.SCALES[0]  # image_width, image_height
        train_max_size = cfg.TRAIN.MAX_SIZE
        aspect_ratio = train_size[0] * 1.0 / train_size[1]
        train_affine_mode = cfg.TRAIN.AFFINE_MODE

        # scale, rotate, flip
        scale_factor = cfg.TRAIN.SCALE_FACTOR
        rotation_factor = cfg.TRAIN.ROT_FACTOR
        flip = cfg.TRAIN.USE_FLIPPED

        # MSPN halfbody
        use_half_body = cfg.TRAIN.USE_HALF_BODY
        prob_half_body = cfg.TRAIN.PRO_HALF_BODY
        num_half_body = cfg.TRAIN.NUM_HALF_BODY
        upper_body_ids = cfg.TRAIN.UPPER_BODY_IDS
        x_ext_half_body = cfg.TRAIN.X_EXT_HALF_BODY
        y_ext_half_body = cfg.TRAIN.Y_EXT_HALF_BODY

        transform = T.Compose(
            [
                T.Convert(aspect_ratio, train_max_size),
                T.HalfBody(use_half_body, num_half_body, prob_half_body, upper_body_ids,
                           x_ext_half_body, y_ext_half_body),
                T.Scale(scale_factor),
                T.Rotate(rotation_factor),
                T.Flip(flip),
                T.CropAndResizeCV2(train_size, train_affine_mode),
                T.ToTensor(),
                T.CropAndResize(train_size, train_affine_mode),
                T.Normalize(pixel_mean, pixel_std, mode=image_format),
                T.GenerateTarget(target_type, sigma, prob_size, train_size),
            ]
        )

    else:
        test_size = cfg.TEST.SCALE  # image_width, image_height
        test_max_size = cfg.TEST.MAX_SIZE
        aspect_ratio = test_size[0] * 1.0 / test_size[1]
        test_affine_mode = cfg.TEST.AFFINE_MODE

        transform = T.Compose(
            [
                T.Convert(aspect_ratio, test_max_size),
                T.CropAndResizeCV2(test_size, test_affine_mode),
            ]
        )
    return transform
