from roboflow import Roboflow

rf = Roboflow(api_key="GOXnl6c0kHfI9Gfr1nns")
project = rf.workspace("mouad-bouhadiba").project("pothole-damage-road-gloev")
dataset = project.version(1).download(model_format="coco",location='./uploads/Pothole Damage Road.v5i.yolov11')
