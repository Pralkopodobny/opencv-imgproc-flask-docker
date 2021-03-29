from flask import Flask, send_file
from flask_restful import Api, Resource, reqparse
import werkzeug
import os
import io

import imageprocessor as improc

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {"data": 'Hello World!'}

class LoadImage(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args['image']
        image_file.save("newImage.jpg")
        improc.to_grayscale("newImage.jpg")
        return {'id': "newImage"}

    def get(self):
        file_path = os.path.join(app.root_path, "newImage.jpg")
        return_data = io.BytesIO()
        with open(file_path, 'rb') as fo:
            return_data.write(fo.read())
        return_data.seek(0)
        os.remove(file_path)
        return send_file(return_data, mimetype='image/jpg', attachment_filename='image.jpg')


api.add_resource(HelloWorld, "/hello")
api.add_resource(LoadImage, "/image")

if __name__ == "__main__":
    app.run(debug=True)
    