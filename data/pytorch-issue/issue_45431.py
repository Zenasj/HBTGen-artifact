import numpy as np
from caffe2.python import core, workspace

workspace.ResetWorkspace()
op = core.CreateOperator("Slice", ["X"], ["Y"], starts=(0, -2), ends=(-1, -1))
workspace.FeedBlob("X", np.array([[1,2,3,4],[5,6,7,8]]))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

import numpy as np
from caffe2.python import core, workspace

workspace.ResetWorkspace()
op = core.CreateOperator("Slice", ["X"], ["Y"], starts=(0,), ends=(1,))
workspace.FeedBlob("X", np.array([[1,2,3,4],[5,6,7,8]]))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))