from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastai.vision.all import *
import uvicorn
import asyncio
import aiohttp
import aiofiles

from io import BytesIO
import sys
import base64
import re
from PIL import Image
# Fastai start
path = Path(__file__).parent
# REPLACE THIS WITH YOUR URL
export_url = "https://www.dropbox.com/s/9p1omxq9d275r8e/export.pkl?dl=1"
export_file_name = 'export.pkl'


def label_func(fn):

    return path/'Maske'/f'{fn.stem}_P.png'


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                f = await aiofiles.open(dest, mode='wb')
                await f.write(await response.read())
                await f.close()


async def setup_learner():
    await download_file(export_url, path / export_file_name)
    try:

        learn = load_learner(path/export_file_name)
        learn.dls.device = 'cpu'
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


def runPredict(base64image, learn):
    image = base64toimage(base64image)
    prediction = learn.predict(image)  # Run prediction, return tensorflow
    plotimage = tensor2image(prediction)  # Create a image plot  of the prediction
    try:
         # Convert the recieved base64string to a image, returns image
        # Analyses the tensor and calculates level ,returns level
        coffeeLevel = checkLevel(prediction)
        #print(f"coffeeLevel: {coffeeLevel}%")
        return {"name": "fastai", "level": coffeeLevel, "image": plotimage}

    except:
        coffeeLevel = "Read error"
        return {"name": "fastai", "level": coffeeLevel, "image": plotimage}


def base64toimage(baseInput):
    try:
        base64_data = re.sub('^data:image/.+;base64,', '', baseInput)
        byte_data = base64.b64decode(base64_data)
        image_data = BytesIO(byte_data)

    except:
        pass

    # print(baseInput)

    lastImgName = ''
    try:
        img = Image.open(image_data)

        t = time.time()

        imagename = 'incommingImage.png'
        lastImgName = os.path.join(path, imagename)
        img.save(lastImgName)
    except:
        pass
    return lastImgName


def tensor2image(tensors):
    ### Saving plot to PNG #####
    # plt.figure(figsize=(30,30))

    plt.imshow(tensors[1])
    filename = 'predictionPlot.png'
    plt.rcParams['axes.facecolor'] = 'black'

    plotfile = os.path.join(path, filename)
    plt.axis('off')

    plt.savefig(plotfile, bbox_inches="tight", pad_inches=0)
    encoded = f'data:image/png;base64,{base64.b64encode(open(plotfile, "rb").read()).decode()}'

    return encoded
# LEvelchecks


def findContainerEdges(slices):
    # print((slices[100]))
    i = 100
    j = 0
    k = 199
    l = 199
    try:
        while slices[i][j] == 0:  # Looks for the first non-black pixel from the left
            leftEdge = j
            j += 1

        while slices[i][k] == 0:  # Looks for the first non-black pixel from the right
            rightEdge = k
            k -= 1

        # Looks for the first white pixel from the bottom
        m = int(((rightEdge-leftEdge)/2) + leftEdge)
        while slices[l][m] != 255:
            bottomEdge = l
            l -= 1
        return {"leftEdge": leftEdge+10, "rightEdge": rightEdge-10, "bottomEdge": bottomEdge}
    except IndexError:
        print("index error")
        return {"leftEdge": None, "rightEdge": None, "bottomEdge": None}
    except:
        print("General Error")
        return {"leftEdge": None, "rightEdge": None, "bottomEdge": None}


def findCoffeeLevel(lines, edges):
    coffee = 0
    notCoffee = 0
    total = 0
    coffeeLevel = 0
    if edges["leftEdge"] == None or ["rightEdge"] == None:
        return "error"
    leftEdge = edges["leftEdge"]
    rightEdge = edges["rightEdge"]
    bottomEdge = edges["bottomEdge"]
    for i in range(0, bottomEdge):  # bredde
        for j in range(leftEdge, rightEdge):  # Høyde
            if(lines[i][j] == 255):  # Hvit/Kaffe
                coffee = coffee+1
            elif(lines[i][j] == 127):  # Gray/Container
                notCoffee = notCoffee+1
            total = total+1
            if coffee != 0:
                coffeeLevel = round((coffee/(coffee + notCoffee))*100, 1)
    return coffeeLevel


def checkLevel(prediction):
    #print(prediction[0])
    lines = prediction[0]
    edges = findContainerEdges(lines)
    coffeeLevel = findCoffeeLevel(lines, edges)
    #print(f"coffeeLevel: {coffeeLevel}%")
    return coffeeLevel


learn = None
# Fastai slutt
