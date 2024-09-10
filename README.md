# Inpainting Application

This project is an inpainting and image processing application that allows user to remove object, fill in that area using inpainting technique and paints the the building with provided color.

Techniques/Models used:
- Segment Anything Model
- Lama
- Image Segmentation using Text and Image Prompts
- clipseg
- OpenCV

Future Scope:
- Here, the object that is painted with the input colors is building.
We have explicitly mentioned this, but we can change that as per our use case.
We can work of the mask and various other techniques to paint it more realistically.
Also, we can work on removing certain defined obstacles automatically from the foreground.


## Cloning the repo

```bash
git clone this_repo_url
```
this_repo_url: copy paste the repo of this url


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.
Note: If you dod pip install, you will have to download the models, weights manually and place it in the respective folders.
```bash
!pip install -r merged_requirements.txt
```
or (recommended way)
```bash
./setup.sh
```
This will install all the dependencies and download the models, weights needed by our pipeline

## Usage

```python
gradio app.py
```
Run the above command to start the gradio app server interface

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
Suggestions are also welcome.
Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
