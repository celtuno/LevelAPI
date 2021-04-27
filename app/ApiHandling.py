from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import traditional as trad
from fastai.vision.all import *

import uvicorn
import asyncio
import aiohttp

from fastapi.middleware.cors import CORSMiddleware
from machineLearning import * 

from pydantic import BaseModel


class Item(BaseModel):
    image: str
    name: str


app = FastAPI()


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Setup the learner on server start"""
    print("Setup the learner on server start")
    global learn
    '''Receive model from onedrive'''
    '''commented out to increase loading speed when not predicting during development'''
    loop = asyncio.get_event_loop()  # get event loop
    tasks = [asyncio.ensure_future(setup_learner())]  # assign some task
    learn = (await asyncio.gather(*tasks))[0]  # get tasks


@app.post("/fastai/predict")
async def machineLearningPrediction(item: Item):
    print("fastai POST test!")
    #Calling the machinelearning python script
    mlprediction = runPredict(item.image, learn)
    return mlprediction


@app.post("/opencv/predict")
async def traditionalPrediction(item: Item):
    print("openCv POST test!")
    # call something like:
    prediction = trad.predict(item.image)
    return prediction


# Redirecting to azurewebpage if entering fastapi container domain

@app.get("/", response_class=HTMLResponse)
async def wrongPageroot():
    '''GET for browser entry to /fastai/predict, giving link to coffeeefinder webpage'''
    print("GET test")
    return getPage()


@app.get("/fastai/predict", response_class=HTMLResponse)
async def wrongPage():
    '''GET for browser entry to /fastai/predict, giving link to coffeeefinder webpage'''
    print("GET test")
    return getPage()


@app.get("/opencv/predict", response_class=HTMLResponse)
async def wrongPage():
    '''GET for browser entry to /fastai/predict, giving link to coffeeefinder webpage'''
    print("GET test")
    return getPage()


def getPage():
    '''GET for browser entry to /fastai/predict, giving link to coffeeefinder webpage'''

    return """
    <html>
        <head>
            <title>Wrong place!</title>
            <meta http-equiv="refresh" content="10; URL=http://coffeefinder.azurewebsites.net/" />
            <link
            rel="stylesheet"
            href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css"
            />
            <link
            href="https://fonts.googleapis.com/css?family=Montserrat"
            rel="stylesheet"
            type="text/css"
            />

            <link
            href="https://fonts.googleapis.com/css?family=Lato"
            rel="stylesheet"
            type="text/css"
            />
            <style>
            body {
            font: 400 15px Lato, sans-serif;
            line-height: 1.8;
            color: #818181;
            }
            h2 {
            font-size: 24px;
            text-transform: uppercase;
            color: #303030;
            font-weight: 600;
            margin-bottom: 30px;
            }
            h4 {
            font-size: 19px;
            line-height: 1.375em;
            color: #303030;
            font-weight: 400;
            margin-bottom: 30px;
            }
            .jumbotron {
            background-color: #f4511e;
            color: #fff;
            padding: 100px 25px;
            font-family: Montserrat, sans-serif;
            }
            </style>

        </head>

        <body>
             <div class="jumbotron text-center">
                <h1>Wrong page</h1>
                <p>Redirecting to the Coffeefinder webpage @ <a href="http://coffeefinder.azurewebsites.net/">coffeefinder.azurewebsites.net</a>!</p>
            </div>
        </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
