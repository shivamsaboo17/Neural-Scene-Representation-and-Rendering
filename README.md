# Neural-Scene-Representation-and-Rendering
Generative Query Network for rendering 3D scenes from 2D images


> There is more than meets the eye when it comes to how we understand a visual scene: our brains draw on prior knowledge to reason and to make inferences that go far beyond the patterns of light that hit our retinas. For example, when entering a room for the first time, you instantly recognise the items it contains and where they are positioned. If you see three legs of a table, you will infer that there is probably a fourth leg with the same shape and colour hidden from view. Even if you can’t see everything in the room, you’ll likely be able to sketch its layout, or imagine what it looks like from another perspective.
    
 [Deepmind](https://deepmind.com/blog/neural-scene-representation-and-rendering/)   

## Learning to see
Making machines understand the scene is a very challenging task for AI. This is implementation of a neural network which is capable of rendering 3D scenes using just a few 2D images. In general case, it may be impossible to predict arbritary view of the scene from finite set of observations, due to the fact that objects occlude themselves and each 2D observation has finite coverage of 3D scene. To address this issue, DeepMind came up with framework of conditional generative modelling to train powerful stochastic generators. Here is an example demonstrating the algorithm's ability to reconstruct 3D scenes with handful of 2D observations.</br>
![](extras/gif_1.gif)
