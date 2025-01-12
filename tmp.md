I have this python function to plot images and their corresponding labels. 
```python
def show_images(images: torch.Tensor,
                labels: Optional[Union[List[str], torch.Tensor]] = None,
                correct_match: Union[list[bool], None] = None,
                save_fig: bool = False,
                filename: str = "images", 
                show: bool = True):
    """
    Displays a batch of images in a grid, optionally with labels and match indicators.

    :param images: Tensor of images with shape (batch_size, channels, height, width).
    :param labels: List of labels for each image, or None if no labels are provided.
    :param correct_match: List of booleans indicating match correctness, or None.
    :param save_fig: Whether to save the figure as an image file. Defaults to False.
    :param filename: Name of the file to save the figure as. Defaults to "images".
    """
    images = images.float()

    # Ensure the image tensor is in the correct shape
    if images.size(3) == 3:  # If channel is last, move it to first
        images = images.permute(0, 3, 1, 2)

    # Normalize and create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True)
    npimg = grid_img.permute(1, 2, 0).numpy()  # Rearrange to (H, W, C) format

    # Normalize labels
    if isinstance(labels, torch.Tensor):
        labels = [str(label.item()) for label in labels]
    elif labels is None:
        labels = []

    # Plot the images
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(npimg)
    ax.axis("off")

    if labels:
        n_images = images.size(0)  # Number of images in the batch
        nrow = 8  # Number of images per row in the grid
        img_size = images.size(2) + 2  # Adjust based on padding

        for i in range(n_images):
            row = i // nrow
            col = i % nrow
            x_pos = col * img_size + img_size // 2 + 1
            y_pos = row * img_size

            label_text = labels[i] if i < len(labels) else "No Label"
            color = (
                "white" if correct_match is None
                else ("green" if correct_match[i] else "red")
            )

            # Display label above each image
            ax.text(
                x_pos, y_pos, label_text,
                color=color,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
            )

    plt.tight_layout()
    if save_fig:
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
```
I want to add an optional numpy array parameter to include a number next to the label. This number is a number between 0 and 1 that can be the confidence of the model for the label or the fraction of noise that caused the prediction to be wrong. The idea is similar to the following.
```python
label_text = f"{labels[i]} : {proportion_array[i]:.2%}" if i < len(labels) else "No Label"
```
MODIFY THE FUNCTION TO INCLUDE THE PROPORTION ARRAY WITHOUT BREAKING THE CURRENT FUNCTIONALITY.
FIND A GOOD NAME FOR THE NEW PARAMETER IF POSSIBLE.