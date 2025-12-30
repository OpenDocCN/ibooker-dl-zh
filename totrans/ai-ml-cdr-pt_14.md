# Chapter 13\. Hosting PyTorch Models for Serving

In the earlier chapters of this book, we looked at many scenarios for training ML models, including those for computer vision, NLP, and sequence modeling. But that was just the first step—a model is of little use without a way for other people to use its power! In this chapter, we’ll take a brief tour of some of the more popular tools that allow you to give them a way to do that.

You should note that taking a trained PyTorch model to a production-ready service will involve a lot more than just deploying it, and that the machine learning operations (MLOps) discipline is designed with that in mind. When you get into the world of serving these models, you’ll need to understand new challenges, such as handling real-time requests, managing computational resources, ensuring reliability, and maintaining performance under varying loads.

Ultimately, MLOps is about bridging the gap between data science and software engineering. That’s beyond the scope of this chapter, but there are some great books about it out there from O’Reilly, including [*Implementing MLOps in the Enterprise*](https://learning.oreilly.com/library/view/implementing-mlops-in/9781098136574)by Yaron Haviv and Noah Gift and [*LLMOps*](https://learning.oreilly.com/library/view/llmops/9781098154196) by Abi Aryan.

This chapter will introduce two popular approaches to serving PyTorch models in production environments.

We’ll begin with TorchServe, the official serving solution from PyTorch, which provides a robust framework that’s designed specifically for serving deep-learning models at scale. TorchServe offers out-of-the-box solutions for standard serving requirements like model versioning, A/B testing, and metrics collection. It’s also an excellent choice for teams who are looking for a production-ready solution with minimal setup.

Then, we’ll explore how to build serving solutions using the popular Flask framework, which is for developers who need more flexibility or have more straightforward serving requirements. Flask’s simplicity and extensive ecosystem also make it an excellent choice for smaller-scale deployments and proof-of-concept services.

As you work through the chapter, you’ll take a hands-on approach in which you’ll take some of the models you created in earlier chapters, let us walk you through how to deploy them, and then call their hosting servers for inference.

# Introducing TorchServe

*TorchServe* is PyTorch’s default serving framework that’s designed for performance and flexibility. You can find it at [*pytorch.org/serve*](http://pytorch.org/serve).

TorchServe’s goal was originally to be a reference implementation on how to properly serve models with a modular extensible architecture, but it has grown beyond that into a fully performant professional-grade framework that is likely more than enough for any serving needs.

It’s also built on a modular architecture with the aim to handle the complexity of serving models at scale. To this end, it’s built from the following key components:

The model server

The PyTorch model server is the central component that handles the lifecycle of models and handles all inference requests. It provides the endpoints for model management and inference, supporting REST and gRPC calls.

Model workers

These are independent processes that load models and perform inference on them. Each one is isolated, so in a multimodel serving environment, they are designed to continue operating should issues in one model arise.

Frontend handlers

These are custom Python classes that handle preprocessing, inference, and post-processing for specific model types. As you’ll see in a moment, when we get hands-on, frontend handlers are complementary to the training code for the model, and it’s good practice to create separate handler classes.

The model store

Model serving in PyTorch uses “model archives,” or MAR files, for the servable object. Once you’ve trained your model, you’ll convert it into this format. The model store is where these are kept.

You can see the architecture diagram for a TorchServe system in [Figure 13-1](#ch13_figure_1_1748549772551219).

![](assets/aiml_1301.png)

###### Figure 13-1\. The TorchServe server infrastructure

For inference, the client application will call the server infrastructure over REST (via default port 8080) or gRPC (via default port 7070). It also provides management endpoints at 8081 and 8082, respectively.

The endpoint will then call the core model server, which in turn will spawn the appropriate number of model workers. The workers will then interface with the model handlers to do preprocessing, inference, and postprocessing. The model itself will be in the model store, and should it not be in memory, the model server will use the request queue to load it.

In the next section, we’ll explore setting up a TorchServe server and using it to provide inference for the first model you created in [Chapter 1](ch01.html#ch01_introduction_to_pytorch_1748548870019566).

# Setting Up TorchServe

I find it’s easiest to learn something if I go through it step-by-step with a simple but representative scenario. So, for TorchServe, we’ll install the environment first and work from there.

## Preparing Your Environment

I strongly recommend using a virtual environment, and I’ll be stepping through this chapter using venv, which is a freely available one from the Python community that you can find [in the Python documentation](https://oreil.ly/FNsPt).

Even if you’ve used virtual environments before, I’d recommend starting with a clean one to ensure you are going to install and use the right set of dependencies.

You can create a virtual environment like this:

```py
python3 -m venv chapter12env
```

And once you’ve done that, you can start it with this:

```py
source chapter12env/bin/activate
```

Then, you’ll be ready to install PyTorch. To get started, I recommend installing TorchServe, the model archiver, and the workflow archiver like this:

```py
pip install torchserve torch-model-archiver torch-workflow-archiver
```

You aren’t limited to these, and you’ll often need to install other dependencies. One of the difficulties of working with TorchServe is that errors may be buried in log files, so it’s hard to figure out which dependencies you’ll also need. At a minimum, I’ve found that starting from a clean system, I also needed to install a JDK after version 11 and PyYAML. Your mileage may vary.

Once you’ve done this, you can change to the directory where you want to work, and within that, you can create a subdirectory called `model_store`:

```py
mkdir model_store
```

With that in place, the first thing you’ll need to do is set up the configuration file for your PyTorch server. It will be a file called *config.properties*.

## Setting Up Your config.properties File

There’s a lot going on in this file, and you can learn more about it on the [PyTorch site](https://oreil.ly/czQh5). But the important settings are the inference and management addresses, which are set to 0.0.0.0 and 8080/8081, respectively (similar to what’s shown in [Figure 13-1](#ch13_figure_1_1748549772551219)). Also, it’s important to set the directory of the model store to be the one you just created. I’ve also set it to log debug messages to help catch dependency issues:

```py
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=model_store
default_response_timeout=120
default_workers_per_model=1
log_level=DEBUG
```

Make sure these settings are in a file with the name *config.properties*.

## Defining Your Model

In [Chapter 1](ch01.html#ch01_introduction_to_pytorch_1748548870019566), you created a model that learned the relationship between two sets of numbers, when the equation describing this relationship was *y* = 2*x* – 1\. When you’re getting the model ready for serving, it’s best practice to have a separate file for the model definition and the model training. The model definition file will then be used in another Python file called the *handler*, which you’ll see momentarily. So, to that end, you should create a model definition file, like this:

```py
import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

Then, save this into a file called *linear.py* so that you can then load and train it with code like the following. Also note the `import` for `SimpleLinearModel` that is bolded:

```py
import torch
import torch.nn as nn
import torch.optim as optim
from linear import SimpleLinearModel

def train_model():
    model = SimpleLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    xs = torch.tensor([[–1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], 
                      dtype=torch.float32)
    ys = torch.tensor([[–3.0], [–1.0], [1.0], [3.0], [5.0], [7.0]], 
                      dtype=torch.float32)

    for _ in range(500):
        optimizer.zero_grad()
        outputs = model(xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()

    return model
# Save model
model = train_model()
torch.save(model.state_dict(), "model.pth")
```

In particular, note the last line, where, after training the model, it’s saved as a *model.pth* file, along with this piece of code: `mode.state_dic()`, which saves out the *state dictionary* (aka the current state of the trained model). I find that this way of saving out a model works best with TorchServe.

Then, you run this code to get the file. You’ll use this later to create a *.mar* file that goes in the model store. We’ll look at that shortly, but first, we’ll need a handler file. You’ll explore that next.

## Creating the Handler File

Once you’ve trained and saved the model, you’ll need to create the handler file. It’s the job of the handler to do the heavy lifting of serving inference—it loads your model, handles data preprocessing, does the inference, and handles any postprocessing to turn the inferred values back into data your users may want.

This file should inherit from the `base_handler` torch class like this:

```py
from ts.torch_handler.base_handler import BaseHandler
```

Once you’ve done this, you’ll need to create a model handler class that overrides this base class and implements a number of methods.

Let’s start with the class declaration and initialization. It’s pretty straightforward: just reporting on class initialization:

```py
class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        logger.info("ModelHandler initialized")
```

Note that an `initialized` property is set to `False` in this code. That may seem odd, but the idea here is that this code will initialize the *class* but it won’t be ready to use until the `initialize` custom function is called. That function will then load the model and get it ready for inference, and at that point, we will set `self.initialized` to be `True`. Note that the model is initialized as a `SimpleLinearModel`, so you’ll need to import it in the same way as you did the training code.

Here’s the code:

```py
def initialize(self, ctx):
    self.manifest = ctx.manifest
    properties = ctx.system_properties
    model_dir = properties.get("model_dir")

    # Load model
    serialized_file = "model.pth"
    model_pt_path = os.path.join(model_dir, serialized_file)
    self.device = torch.device("cuda:" + str(properties.get("gpu_id")) 
                  if torch.cuda.is_available() else "cpu")

    # Initialize model
    self.model = SimpleLinearModel()
    state_dict = torch.load(model_pt_path, weights_only=True)
    self.model.load_state_dict(state_dict)
    self.model.to(self.device)
    self.model.eval()

    self.initialized = True
    return self
```

It begins by reading `ctx.manifest`—which is information in the *MAR* file (such as the model directory) that lets you use the model. We’ll see how to create that file shortly.

The rest of the code is pretty straightforward: we create an instance of the model (called `SimpleLinearModel`, as shown in the previous code) and load its weights from where they were saved in (*model.pth*). We can then push the model to the device, where we will do inference, such as the CPU or CUDA, if it’s available. Then, we’ll put the model into evaluation mode for inference. I have found that TorchServe works best if you save the model with its state dictionary, so be sure to load that back, as shown.

At that point, we set `self.initialized` to `True`, and we’re good to go for inference!

The next method we need to override is the *preprocess method*, which will take the input data and turn it into the format the model needs. There are many ways in which you could post the data to the backend server, and it’s in the preprocess function that you’d handle them. You could, for example, just take basic parameters, or you could allow your user to post a JSON file. It’s up to you, and this flexibility in the TorchServe architecture opens up those possibilities.

For the sake of simplicity, I’m just going to take a basic parameter in this code:

```py
def preprocess(self, data):
    # Get the value directly without decoding
    value = float(data[0].get("body"))
    tensor = torch.tensor([value], dtype=torch.float32).view(1, 1)
    return tensor.to(self.device)
```

As you can see, the main purpose of this is not only to *get* the parameters but also to *reformat* them into the format the model needs. So, it gets the parameter from the request call and turns it into a single-dimension tensor, which it will then return. This tensor will be used in the inference method.

The next method to override is the inference method, and here’s where we will pass the data into the model and get its response back. Here’s the code:

```py
def inference(self, data):
    """Run inference on the preprocessed data"""
    with torch.no_grad():
        results = self.model(data)
    return results
```

This receives the preprocessed data from the previous step, so we can just get the output by passing in this data. We run it with `torch.no_grad()` because we aren’t interested in backpropagation, just a straight inference.

Finally, there’s postprocessing. The end user isn’t expecting a tensor back, but they are expecting more human-readable data, so we do the reverse of the preprocess step and cast the result back to a float by using NumPy:

```py
def postprocess(self, inference_output):
    """Return inference result"""
    return inference_output.tolist()
```

These steps are, as you can see, very customized to this model, and I’ve deliberately made them bare-bones so you can see the flow—but the general architecture remains the same for other models, regardless of the complexity. The goal is to give you a standard way to approach the problem, and code that extends the `BaseHandler` like this makes it easy for you to take advantage of all the unseen aspects of the server infrastructure—not least, passing the data around!

## Creating the Model Archive

Starting with a trained model, you can create the archive file for it by calling `torch-model-archiver`, which is a command-line tool that’s provided as part of TorchServe.

There are a few things you need to note and be careful of here. Here’s the command, and I’ll discuss the parameters afterward:

```py
torch-model-archiver 
  --model-name simple_linear
  --version 1.0
  --serialized-file model.pth
  --handler model_handler.py
  --model-file models/linear.py
  --export-path model-store
  --force
  --extra-files models/linear.py,models/__init__.py
```

Before running it, make sure you have a *model-store* (or similar) directory that you will store the archived model in. Don’t mix it up with your source code! This can simply be a subdirectory, and you’ll specify that directory in the `export-path` parameter.

The `model-name` parameter will specify the name of the model in the model store. You can call it whatever you want—it doesn’t have to be the class name. In this case, I called it `simple-linear`.

The `version` parameter can be whatever you want, and you’ll use it for tracking. As you train new versions of your model for new and different scenarios or bug fixes, you’ll likely want to keep track of which version does what. You can let the server know about that here.

When you trained the model, you saved the weights and the state dictionary out to a file. You specify this file with the `serialized-file` parameter. In this case, it’s `model.pth`.

The handler file is specified with the `handler` parameter.

A common problem I have encountered occurs when the model training file contains the model definition *and* the handler also contains it. TorchServe gets confused as to where to get the model definition, and that’s why I separated it out in this case. But if you need to do that, it’s important to specify *where* the model definition is, and you can do that with the `model-file` parameter. Note that following this methodology, you should also point to the location of the model file by using the `extra-files` parameter. To make the model file importable to both the model training and model handler as a package, I put the file in a directory called *models* and put an empty file called *_ _init_ _.py* in there. To make sure that the model archiver uses these, *both* of these files are specified in the `extra-files` parameter.

The `force` parameter just overwrites any existing model archive in the *model-store* directory. When you’re learning, this parameter saves you from having to delete the model archive manually when trying different things. However, in a production system, you should use it carefully!

Once this line runs correctly, your *model-store* directory should contain a *simple_linear.mar* file.

## Starting the Server

Once you’ve created your archive, you can start TorchServe and have it load that model. Here’s an example command:

```py
torchserve
  --start
  --model-store model-store
  --ts-config config/config.properties
  --disable-token-auth
  --models simple_linear=model_store/simple_linear.mar
```

The `start` parameter instructs TorchServe to start. You can also use `--stop` in the same way to stop it from executing (and filling your terminal with a wall of text).

The `model-store` parameter should also point to the directory where you store the *.mar* file that you created earlier.

The `ts-config` parameter should point to the configuration properties file you created earlier.

When you’re learning and testing, you can use `--disable-token-auth` so that commands you send to the server to test your models don’t need authentication. However, in proper production systems, you probably wouldn’t want to use it!

The `models` parameter is a list of models that you want the server to make available to your users. In this case, there’ll just be one, and it’s the `simple_linear` model we defined. If you give the path to this model as the value, you’ll see that it’s the location of the *mar* file in the `model_store`.

If all goes well, you should then see TorchServe start in your terminal and give you a wall of status text, a little like in [Figure 13-2](#ch13_figure_2_1748549772551268).

Note that if this text is constantly scrolling, it’s likely that there was an error in starting up TorchServe. From experience, I would say that there are dependencies that TorchServe needs (like PyYAML) that haven’t been installed. If that’s the case, then the configuration file was set to debug, and you can inspect the *models_log.log* file in the *logs* directory to see what’s going on.

You may also see errors like the one in [Figure 13-2](#ch13_figure_2_1748549772551268), where it can’t find the `nvgpu` module. This module is used by TorchServe to do GPU-based inference with an Nvidia GPU. Because I was running on a Mac in this case, you can safely ignore the error, and all inference will just run on the CPU, as per the code in the handler.

![](assets/aiml_1302.png)

###### Figure 13-2\. Starting TorchServe

## Testing Inference

Once the server is successfully up and running, you can test it by using `curl` from another terminal.

So, to get an inference from the model, you can `curl` like this:

```py
curl -X POST http://127.0.0.1:8080/predictions/simple_linear -H 
                       "Content-Type: text/plain" -d "5.0"
```

Notice that it is an HTTP POST to the predictions endpoint. We specify the `simple_linear` model name as defined with the *.mar* file earlier, and we can then add the header (with the `-H` parameter) as plain text containing the data `5.0`.

As you may recall, the model learned the linear relationship *Y* = 2*x* – 1, so in this case, *x* will be 5 and the result will be a number close to 9.

The return should look like this:

```py
[
  8.997674942016602
]
```

Your value may vary, based on how your model was trained, but it should be a value very close to 9.

You can also use the management endpoint to inspect the models that the server is hosting, like this:

```py
curl http://localhost:8081/models
```

The response will contain the name and location of the *.mar* file for each model on the server:

```py
{
  "models": [
    {
      "modelName": "simple_linear",
      "modelUrl": "model_store/simple_linear.mar"
    }
  ]
}

```

Note that earlier, when you did inference on the model, you used the `predictions` endpoint followed by a model name, which in that case was `simple_linear`. This key should map to a model name in this models collection, or you would have gotten an error.

Finally, if you want to explore the details of a specific model, you can call the management URL (via port 8081, as earlier) with the model’s endpoint and the name of the model you want to know more about:

```py
curl http://localhost:8081/models/simple_linear
```

The server will return some detailed specs on the model, along with information you can use to help debug any issues. Here’s an example:

```py
[
  {
    "modelName": "simple_linear",
    "modelVersion": "1.0",
    "modelUrl": "model_store/simple_linear.mar",
    "runtime": "python",
    "minWorkers": 1,
    "maxWorkers": 1,
    "batchSize": 1,
    "maxBatchDelay": 100,
    "responseTimeout": 120,
    "startupTimeout": 120,
    "maxRetryTimeoutInSec": 300,
    "clientTimeoutInMills": 0,
    "parallelType": "",
    "parallelLevel": 0,
    "deviceType": "gpu",
    "continuousBatching": false,
    "useJobTicket": false,
    "useVenv": false,
    "stateful": false,
    "sequenceMaxIdleMSec": 0,
    "sequenceTimeoutMSec": 0,
    "maxNumSequence": 0,
    "maxSequenceJobQueueSize": 0,
    "loadedAtStartup": true,
    "workers": [
      {
        "id": "9000",
        "startTime": "2024-11-16T08:26:59.394Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 20395,
        "gpu": true,
        "gpuUsage": "failed to obtained gpu usage"
      }
    ],
    "jobQueueStatus": {
      "remainingCapacity": 1000,
      "pendingRequests": 0
    }
  }
]
```

Notice, for example, that the `deviceType` is expecting `gpu`. However, since I don’t have `nvgpu` on this system (see the earlier note about when you ran the server), it can’t load from the GPU, and the workers reported on that. It’s OK for that to be the case in my dev box, which doesn’t have the Nvidia GPU—but should you be running this on a server that *does* have the Nvidia GPU, that message is something you’d want to follow up on, and it’s likely an issue in your handler file.

## Going Further

The foregoing was a bare-bones example to help you understand the nuts and bolts of how TorchServe works. As you use more sophisticated models that take in more complex data, the basic pattern you followed here should follow suit. In particular, the breakdown of preprocessing, inference, and postprocessing in the handler file is an enormous help! Additionally, the PyTorch ecosystem has add-ons for common scenarios to help you avoid having to write preprocessing code to begin with! For example, if you are interested in image classification and are worried about taking an image and turning it into tensors so that you can do inference on that image, the `ImageClassifier` class builds on the base handler to do this for you, and you can have image classification without needing to write a preprocessor. To see more of this in action, take a look at the open source examples at the PyTorch repository. In particular, you can go to [this GitHub page](https://oreil.ly/YA4v4) for an example of how to create a handler for MNIST images.

You’ll find many more useful examples in that repo, but I would still recommend going through the steps to get a bare-bones example like the one we showed here up and running first. There are a lot of steps and a lot of concepts, and it’s easy to get lost in the maze.

# Serving with Flask

While TorchServe is extremely powerful, a great alternative that’s super easy to use is Flask. Flask is a lightweight and flexible web framework for Python that enables efficient development of web applications and APIs.

You can use Flask to build everything from minimal single-endpoint services to complex web applications, starting with just a few lines of code. It perfectly complements PyTorch by giving you the ability to host models for inference, as we’ll explore in this section.

As a microframework, Flask provides the essential components for web development—routing, request handling, and templating—while allowing you to select additional functionality as needed. It is highly extensible, with stuff like a backend database, authentication, etc., and there’s also a vibrant ecosystem of extensions available to use off-the-shelf. Because of all of this, Flask has become a standard tool in web development, powering applications across industries at all scales.

In this chapter, we’ll just explore Flask from the hosting model’s perspective, but I’d encourage you to dig deeper into the framework if you’re interested in serving Python code—not just PyTorch!

Now, let’s take a look at the same example that we used for TorchServe. This will help you see how simple Flask makes serving applications!

## Creating an Environment for Flask

First, to use Flask, you’ll need to install it and any dependencies. If you’ve been using the same environment as earlier in this chapter, you can simply update it with this:

```py
pip install flask
```

Then, just ensure that you have a model definition and training file—exactly the same as those from earlier in this chapter—and that you have trained a model and saved it with its state dictionary as a file called *model.pth*.

With those in hand, all you’ll need is a single Python file that I’m going to call *app.py*, which will be the primary server application that Flask will use. We’ll explore the code in that file next.

## Creating a Flask Server in Python

To create a server with Flask, you implement an app that creates a new Flask instance by using the following code:

```py
app = Flask(__name__)

```

Then, on the app object, you can specify routes, such as `predict` for doing prediction on models. To serve, you implement the code for the endpoints at these routes. Here’s an example of a full Flask server for our simple app:

```py
from flask import Flask, request, jsonify
import torch
from model_def import SimpleLinearModel

app = Flask(__name__)

# Load the trained model
model = SimpleLinearModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    value = float(request.form.get('value', 0))
    input_tensor = torch.tensor([[value]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)

    return jsonify({
        "input": value,
        "prediction": prediction.item()
    })

if __name__ == "__main__":
    app.run(port=5001)
```

###### Note

Flask documentation and samples tend to use port 5000\. If you’re using a Mac as your development box, you might have issues with this as it conflicts with the port used by Airplay. To that end, I’ve used 5001 in this sample.

In this case, we declare the `SimpleLinearModel` and load it along with its state dictionary. Then, we put it into `eval()` mode to get it ready for inference.

Then, it becomes as simple as creating a route that we call `predict` and then implementing the inference within it. As you can see, we handle getting the `value` from the HTTP `POST`, converting it into a tensor, and getting the prediction back when we pass that tensor to the model.

To make the return a little friendlier, I used `jsonify` to turn it into a name-value pair. As you can see, that’s much simpler than using TorchServe, but for that simplicity, you give up power. There’s no set of base classes to handle preprocessing, post-processing, etc., and if you want to scale or implement multiple worker threads, you’ll have to do it yourself.

I think this is a really useful and powerful server mechanism for smaller-scale environments, as well as for learning how to serve. For large-scale production environments, it’s a lot of extra work, but it can definitely handle the load.

For inference, you can `curl` a POST to the model like this:

```py
curl -X POST -d "value=5" http://localhost:5001/predict
```

And the response will be the JSON payload:

```py
{"input":5.0,"prediction":8.993191719055176}
```

In addition to TorchServe and Flask, there are many other serving options, such as ONNX and FastAPI.

# Summary

In this chapter, we explored two popular approaches to serving PyTorch models in production environments. You started with TorchServe, which is PyTorch’s official serving solution that offers a robust framework with built-in support for model versioning, A/B testing, and metrics collection. It’s also designed to be highly scalable at runtime, with a worker-thread architecture that’s configurable based on the needs of your app. And while TorchServe requires more setup and understanding of its components like model workers and frontend handlers, I think it’s worth investing the time in rolling up your sleeves and understanding all the different components and how they work together. To that end, you explored step-by-step how to take the simple linear model example from [Chapter 1](ch01.html#ch01_introduction_to_pytorch_1748548870019566), save it, archive it, build a handler for it, and launch the server with the model details.

Then, you explored how to use Flask as a lightweight alternative that’s extremely quick and simple to get up and running. You saw how its minimalist approach makes it ideal for smaller-scale deployments or proof-of-concept services. It’s not limited to those, but as you move to production scale, you’ll likely need to implement more code. Of course, that’s not necessarily a disadvantage, as it gives you more granular control over your serving environment.

Both approaches have their place in the ML serving ecosystem. TorchServe shines in enterprise environments requiring comprehensive model management, while Flask’s simplicity makes it perfect for smaller projects or learning environments with a smooth glide path toward scalable production. Of course, you aren’t limited to just these two, and new frameworks are coming online all the time—in particular, one called FastAPI, which is rapidly growing in popularity. Which one you should choose ultimately depends on your specific needs around scaling, monitoring, and deployment complexity.

Next, in [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797), you’re going to look at third-party models that have been pretrained for you and various registries and hubs that you can load them from.