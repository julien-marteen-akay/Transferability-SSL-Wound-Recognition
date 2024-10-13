import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def plot_gradcam(model, target_layers, dataset, indices=0):
    try:
        iterator = iter(indices)
    except TypeError:
        indices = [indices]
        
    for i in indices:
        input_tensor, target = dataset[i]
        input_tensor = input_tensor[None, ...]  # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        targets = [ClassifierOutputTarget(target)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(input_tensor[0, ...].permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax[0].imshow(visualization)
        ax[1].imshow(input_tensor[0, ...].permute(1, 2, 0).numpy())
        plt.show()


def get_gradcam_image(*, model, target_layers, target, input_tensor: torch.tensor):
    """
    Applies gradcam to a single image and returns the image overlayed with the gradcam heatmap, e.g. for subsequent plotting.
    Use plot_gradcam if you want to plot multiple images.

    Arguments:
        model: The model to be analyzed. Must be differentiable (turn grads on by calling Classifier.unfreeze_backbone) and have the classification head.
        target_layers: A list with layers, most of the times, one is interested in including the deepest layer, which must return feature maps (e.g. conv layers).
        target (int): The index of the target class, aka the index of the label.
        input_tensor (torch.tensor): The input image, either of shape (1 x C x H x W) or (C x H x W). It will be automatically reshaped to be rank 4.

    Returns:
        The input image overlayed with the gradcam heatmap.
    """
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor[None, ...]

    if len(input_tensor.shape) < 4:
        raise ValueError(f"input_tensor has wrong rank. Current shape={input_tensor.shape} but should be a rank 4 tensor.")

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [ClassifierOutputTarget(target)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_tensor[0, ...].permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
    return visualization
