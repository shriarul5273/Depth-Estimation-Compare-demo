# Depth Anything 3 AnySize

## 🔄 Key Modifications from the [Original Repo](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Native-Resolution Inputs:** Images are now processed at their original resolution by default. During inference, inputs are padded to the ViT patch size, and outputs (depth/confidence/sky maps and processed images) are cropped back to the source height and width. Using larger inputs now will increase memory and compute requirements.
- **Updated Defaults:** The CLI defaults to `--process-res None --process-res-method keep`, and the API uses `process_res=None, process_res_method="keep"`. See `docs/CLI.md` and `docs/API.md` for details.
- **Optional Downscaling:** For faster inference and lower memory usage, set `process_res` (e.g., `720`) with a resize strategy like `--process-res-method upper_bound_resize`.
- **Original Baseline:** Previously, images were resized to 504 px on the long side.
- **Implementation Details:** Input padding is handled in `src/depth_anything_3/utils/io/input_processor.py`, and output cropping is managed in `src/depth_anything_3/api.py`.

--------------------------------------
