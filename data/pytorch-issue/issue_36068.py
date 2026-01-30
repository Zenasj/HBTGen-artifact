import os
from flask import Flask, render_template, Response
import multiprocessing as mp

app = Flask(__name__)

@app.route('/video_feed1')
def video_feed1():
    return Response(gen(segPrediction()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    mp.set_start_method('spawn',force=True)
    app.run(host='0.0.0.0', threaded=False, processes=2)

import os
from flask import Flask, render_template, Response
import multiprocessing as mp
mp.set_start_method('spawn',force=True)

app = Flask(__name__)

@app.route('/video_feed1')
def video_feed1():
    return Response(gen(segPrediction()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=False, processes=2)