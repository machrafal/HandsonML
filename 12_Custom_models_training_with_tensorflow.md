{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "99d922f6-9409-4cf8-95ff-0c3fecd405cd",
   "metadata": {},
   "source": [
    "# Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "829a0d07-c40f-40d2-b47f-bf08d3a4b155",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-27 04:43:20.742881: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
      "2024-09-27 04:43:21.066482: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory\n",
      "2024-09-27 04:43:21.066567: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.\n",
      "2024-09-27 04:43:21.115013: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
      "2024-09-27 04:43:22.669902: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory\n",
      "2024-09-27 04:43:22.670112: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory\n",
      "2024-09-27 04:43:22.670127: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "assert sys.version_info >= (3, 5)\n",
    "\n",
    "import sklearn\n",
    "assert sklearn.__version__ >= \"0.20\"\n",
    "\n",
    "try:\n",
    "    # tensorflow_version only exists in Colab.\n",
    "    %tensorflow_version 2.x\n",
    "except Exception:\n",
    "    pass\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "# assert tf.__version__ >= \"2.4\"\n",
    "\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)\n",
    "\n",
    "%matplotlib inline\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "mpl.rc('axes', labelsize=14)\n",
    "mpl.rc('xtick', labelsize=12)\n",
    "mpl.rc('ytick', labelsize=12)\n",
    "\n",
    "PROJECT_ROOT_DIR = \".\"\n",
    "CHAPTER_ID = \"deep\"\n",
    "IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, \"images\", CHAPTER_ID)\n",
    "os.makedirs(IMAGES_PATH, exist_ok=True)\n",
    "\n",
    "def save_fig(fig_id, tight_layout=True, fig_extension=\"png\", resolution=300):\n",
    "    path = os.path.join(IMAGES_PATH, fig_id + \".\" + fig_extension)\n",
    "    print(\"Saving figure\", fig_id)\n",
    "    if tight_layout:\n",
    "        plt.tight_layout()\n",
    "    plt.savefig(path, format=fig_extension, dpi=resolution)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "712b0f01-6891-4846-948e-5f11b2737f36",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.simplefilter('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf31b69c-23b6-4a0d-9c51-02c8751d24c4",
   "metadata": {},
   "source": [
    "## Tensors and operations"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96eaa281-9968-49ff-a2bb-12a50a5271c2",
   "metadata": {},
   "source": [
    "### Tensors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "511836d5-e150-4cc5-9a85-ec692a6ea04d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-27 04:43:25.962526: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory\n",
      "2024-09-27 04:43:25.962616: W tensorflow/stream_executor/cuda/cuda_driver.cc:263] failed call to cuInit: UNKNOWN ERROR (303)\n",
      "2024-09-27 04:43:25.962675: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (jupyter_machrafal): /proc/driver/nvidia/version does not exist\n",
      "2024-09-27 04:43:25.963173: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\n",
      "To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 3), dtype=float32, numpy=\n",
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]], dtype=float32)>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0f29be64-71d5-4c3b-8a55-35a69d14d6c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=100>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.constant(100) # scalar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a0e41565-1e62-4dce-89f3-5f53c2a71c84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 3), dtype=float32, numpy=\n",
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]], dtype=float32)>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t = tf.constant([[1., 2., 3.], [4., 5., 6.]])\n",
    "t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "afbac577-978a-4ebc-9956-2d4f64a2583c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TensorShape([2, 3])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f53ce5d8-f8bc-4361-ae9d-949c22b7c9ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tf.float32"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t.dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bab77ff3-d474-426f-aa8e-764ff9f9d99c",
   "metadata": {},
   "source": [
    "### Indexing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1e29d789-24c8-458f-9869-1cbe782c2ce2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 2), dtype=float32, numpy=\n",
       "array([[2., 3.],\n",
       "       [5., 6.]], dtype=float32)>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t[:, 1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "35c2e10d-0641-44bb-9060-a1701b4d0a65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 1), dtype=float32, numpy=\n",
       "array([[2.],\n",
       "       [5.]], dtype=float32)>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t[..., 1, tf.newaxis]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd570cbc-7b4d-456b-baa0-c3ddfa9c9356",
   "metadata": {},
   "source": [
    "### Ops"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9950295a-e8ca-4c89-a909-9df12967f191",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 3), dtype=float32, numpy=\n",
       "array([[11., 12., 13.],\n",
       "       [14., 15., 16.]], dtype=float32)>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t + 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ae31644c-78fb-47eb-9ab8-631941564ad3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 3), dtype=float32, numpy=\n",
       "array([[ 1.,  4.,  9.],\n",
       "       [16., 25., 36.]], dtype=float32)>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.square(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "49ab471e-0fa4-4620-ad8e-37769826d8b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 3), dtype=float32, numpy=\n",
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]], dtype=float32)>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9843167d-da83-40a7-bf1b-93cc84dbf3b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3, 2), dtype=float32, numpy=\n",
       "array([[1., 4.],\n",
       "       [2., 5.],\n",
       "       [3., 6.]], dtype=float32)>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.transpose(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "65744e85-051a-4332-8ca4-5592204536c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 2), dtype=float32, numpy=\n",
       "array([[14., 32.],\n",
       "       [32., 77.]], dtype=float32)>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t @ tf.transpose(t)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "958ea359-89ff-4fbf-a786-10552766d888",
   "metadata": {},
   "source": [
    "### Using `keras.backend`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "94c0b9b6-3c5c-433d-b13d-a433bcd57b43",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3, 2), dtype=float32, numpy=\n",
       "array([[11., 26.],\n",
       "       [14., 35.],\n",
       "       [19., 46.]], dtype=float32)>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from tensorflow import keras\n",
    "K = keras.backend\n",
    "K.square(K.transpose(t)) + 10"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ca2ab68-4de8-4257-80c9-3828507b65da",
   "metadata": {},
   "source": [
    "### From/To NumPy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0fee9820-e165-442a-b43d-be6f6b977d87",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([2., 4., 5.])\n",
    "tf.constant(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "9b6b3a48-c921-4f34-8ac5-22180623bb4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]], dtype=float32)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t.numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "fc10a2dd-39fb-40c9-a1e4-64f515ea01e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]], dtype=float32)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "37a46c69-840e-44c0-b5e1-e316a21c60af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 4., 16., 25.])>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.square(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2171ce56-3394-4937-b663-0f1b7b88f17b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.,  4.,  9.],\n",
       "       [16., 25., 36.]], dtype=float32)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.square(t)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a0a7037-7fef-426b-818c-8cea4bb74f64",
   "metadata": {},
   "source": [
    "### Conflicting Types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "6ababa82-1ebc-4b75-bd8b-3a378412c1c9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:AddV2]\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    tf.constant(2.0) + tf.constant(40)\n",
    "except tf.errors.InvalidArgumentError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "601dddd9-bc18-45ec-8bec-875f1fd45573",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a double tensor [Op:AddV2]\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    tf.constant(2.0) + tf.constant(40., dtype=tf.float64)\n",
    "except tf.errors.InvalidArgumentError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "28f655ec-ad76-458f-91d5-c9d4a823dbec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=42.0>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t2 = tf.constant(40., dtype=tf.float64)\n",
    "tf.constant(2.0) + tf.cast(t2, tf.float32)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ffe420e2-d13c-4aaf-9c56-a1cddab9710b",
   "metadata": {},
   "source": [
    "### Strings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "7b86d5ff-8037-4d2d-a456-1bf89cbb84a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=string, numpy=b'hello world'>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.constant(b\"hello world\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "efc7f897-4b59-44a2-966d-48ad7a6a02e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=string, numpy=b'caf\\xc3\\xa9'>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.constant(\"café\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3449b741-ccc5-4a84-9cea-2a1476fc9908",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 99,  97, 102, 233], dtype=int32)>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "u = tf.constant([ord(c) for c in \"café\"])\n",
    "u"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "540f278b-0b8d-4cd6-9a0e-6df9dbf9df6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=4>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b = tf.strings.unicode_encode(u, \"UTF-8\")\n",
    "tf.strings.length(b, unit=\"UTF8_CHAR\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "a1080f9c-9fe9-48ba-a4c8-da17df7370da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 99,  97, 102, 233], dtype=int32)>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.strings.unicode_decode(b, \"UTF-8\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06f55f46-528a-48a4-81ab-c6ac5af6c9e6",
   "metadata": {},
   "source": [
    "### String arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "4c899361-691d-448d-b3a3-1b4dee8c0434",
   "metadata": {},
   "outputs": [],
   "source": [
    "p = tf.constant([\"Café\", \"Coffee\", \"caffè\", \"咖啡\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "df8a9cc2-c226-4d65-8b7f-e8118d741db8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(4,), dtype=int32, numpy=array([4, 6, 5, 2], dtype=int32)>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.strings.length(p, unit=\"UTF8_CHAR\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "97ff1216-8e41-428a-bfd0-2766a9f72c55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101],\n",
       " [99, 97, 102, 102, 232], [21654, 21857]]>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r = tf.strings.unicode_decode(p, \"UTF8\")\n",
    "r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "5b0ee745-dee6-4556-9518-8a534fe18102",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101],\n",
      " [99, 97, 102, 102, 232], [21654, 21857]]>\n"
     ]
    }
   ],
   "source": [
    "print(r)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2763d1d8-70ee-4c00-b02b-c8eb6d3307dc",
   "metadata": {},
   "source": [
    "### Ragged tensors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "99a450da-9f2b-402e-8a5c-aced1b2d78ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tf.Tensor([ 67 111 102 102 101 101], shape=(6,), dtype=int32)\n"
     ]
    }
   ],
   "source": [
    "print(r[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "951f08fd-ebea-4b23-a379-50e64863ef02",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<tf.RaggedTensor [[67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232]]>\n"
     ]
    }
   ],
   "source": [
    "print(r[1:3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "d0b3f011-9e1b-4d76-85b9-34587763d80b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101],\n",
      " [99, 97, 102, 102, 232], [21654, 21857], [65, 66], [], [67]]>\n"
     ]
    }
   ],
   "source": [
    "r2 = tf.ragged.constant([[65, 66], [], [67]])\n",
    "print(tf.concat([r, r2], axis=0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "9ff32d75-ef2d-41f2-8fac-6e001f3a1708",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<tf.RaggedTensor [[67, 97, 102, 233, 68, 69, 70], [67, 111, 102, 102, 101, 101, 71],\n",
      " [99, 97, 102, 102, 232], [21654, 21857, 72, 73]]>\n"
     ]
    }
   ],
   "source": [
    "r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])\n",
    "print(tf.concat([r, r3], axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "cdcc761a-5575-4a81-ab0d-b80746016d47",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(4,), dtype=string, numpy=array([b'DEF', b'G', b'', b'HI'], dtype=object)>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.strings.unicode_encode(r3, \"UTF-8\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "66a5af27-a5d8-41b7-95eb-23defe8701e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(4, 6), dtype=int32, numpy=\n",
       "array([[   67,    97,   102,   233,     0,     0],\n",
       "       [   67,   111,   102,   102,   101,   101],\n",
       "       [   99,    97,   102,   102,   232,     0],\n",
       "       [21654, 21857,     0,     0,     0,     0]], dtype=int32)>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r.to_tensor()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8cd65be-7331-40ae-a994-309f3075c095",
   "metadata": {},
   "source": [
    "### Sparse tensors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "b48090d5-871c-4698-b1ad-f8c5a1802cf8",
   "metadata": {},
   "outputs": [],
   "source": [
    "s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],\n",
    "                    values=[1., 2., 3.],\n",
    "                    dense_shape=[3, 4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "80aa4847-8c76-43db-bd1e-45c994c95d2d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SparseTensor(indices=tf.Tensor(\n",
      "[[0 1]\n",
      " [1 0]\n",
      " [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))\n"
     ]
    }
   ],
   "source": [
    "print(s)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "ecb35c9c-5407-4ff0-b4b9-9c23ac76538a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3, 4), dtype=float32, numpy=\n",
       "array([[0., 1., 0., 0.],\n",
       "       [2., 0., 0., 0.],\n",
       "       [0., 0., 0., 3.]], dtype=float32)>"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.sparse.to_dense(s)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "51051888-3f61-4a90-a1a2-a964c898a7a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "s2 = s * 2.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "3985c64b-c5fe-4185-b216-1748555a8c31",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "unsupported operand type(s) for +: 'SparseTensor' and 'float'\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    s3 = s + 1.\n",
    "except TypeError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "6473ee3f-1415-47ec-91b7-c0a58005a21f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3, 2), dtype=float32, numpy=\n",
       "array([[ 30.,  40.],\n",
       "       [ 20.,  40.],\n",
       "       [210., 240.]], dtype=float32)>"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])\n",
    "tf.sparse.sparse_dense_matmul(s, s4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "09e4ec82-91f1-4296-b029-11ee0d92d800",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SparseTensor(indices=tf.Tensor(\n",
      "[[0 2]\n",
      " [0 1]], shape=(2, 2), dtype=int64), values=tf.Tensor([1. 2.], shape=(2,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))\n"
     ]
    }
   ],
   "source": [
    "s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],\n",
    "                     values=[1., 2.],\n",
    "                     dense_shape=[3, 4])\n",
    "print(s5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "0cef3177-b830-40e9-8cd7-13867486a14d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{{function_node __wrapped__SparseToDense_device_/job:localhost/replica:0/task:0/device:CPU:0}} indices[1] = [0,1] is out of order. Many sparse ops require sorted indices.\n",
      "    Use `tf.sparse.reorder` to create a correctly ordered copy.\n",
      "\n",
      " [Op:SparseToDense]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-27 04:43:26.656920: W tensorflow/core/framework/op_kernel.cc:1780] OP_REQUIRES failed at sparse_to_dense_op.cc:162 : INVALID_ARGUMENT: indices[1] = [0,1] is out of order. Many sparse ops require sorted indices.\n",
      "    Use `tf.sparse.reorder` to create a correctly ordered copy.\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    tf.sparse.to_dense(s5)\n",
    "except tf.errors.InvalidArgumentError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "e7f4b6b7-94b1-414b-b917-6e50b93eba57",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3, 4), dtype=float32, numpy=\n",
       "array([[0., 2., 1., 0.],\n",
       "       [0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0.]], dtype=float32)>"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "s6 = tf.sparse.reorder(s5)\n",
    "tf.sparse.to_dense(s6)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28e17c97-9dbe-43d9-be53-3df60502dcf9",
   "metadata": {},
   "source": [
    "### Sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "924094e8-1ca7-4126-8855-260b68b16074",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 6), dtype=int32, numpy=\n",
       "array([[ 2,  3,  4,  5,  6,  7],\n",
       "       [ 0,  7,  9, 10,  0,  0]], dtype=int32)>"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])\n",
    "set2 = tf.constant([[4, 5, 6], [9, 10, 0]])\n",
    "tf.sparse.to_dense(tf.sets.union(set1, set2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "2bdd0d58-cc6b-4249-a520-71f47e09039e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 3), dtype=int32, numpy=\n",
       "array([[2, 3, 7],\n",
       "       [7, 0, 0]], dtype=int32)>"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.sparse.to_dense(tf.sets.difference(set1, set2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "b471c13f-2b69-439a-884c-cf97b5e0299f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2, 2), dtype=int32, numpy=\n",
       "array([[5, 0],\n",
       "       [0, 9]], dtype=int32)>"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.sparse.to_dense(tf.sets.intersection(set1, set2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "799d10f8-ea81-4456-98cd-61122ac2c0d1",
   "metadata": {},
   "source": [
    "### Variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "37f9fcaf-c770-45a1-8cc3-6a2cad3814e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "aa90eed3-59b1-4308-8fd2-0c30f19bd7ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=\n",
       "array([[ 2.,  4.,  6.],\n",
       "       [ 8., 10., 12.]], dtype=float32)>"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "v.assign(2 * v)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "ed643c86-8ebf-4e7f-8691-610ed892e91a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=\n",
       "array([[ 2., 42.,  6.],\n",
       "       [ 8., 10., 12.]], dtype=float32)>"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "v[0, 1].assign(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "0904bb60-39f2-455a-9735-1d27447c2789",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=\n",
       "array([[ 2., 42.,  0.],\n",
       "       [ 8., 10.,  1.]], dtype=float32)>"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "v[:, 2].assign([0., 1.])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "ad771db6-e6b7-4557-b279-e02fffe38448",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "'ResourceVariable' object does not support item assignment\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    v[1] = [7., 8., 9.]\n",
    "except TypeError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "11eb2048-ab4f-48d4-a264-b3bd0e8f36ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=\n",
       "array([[100.,  42.,   0.],\n",
       "       [  8.,  10., 200.]], dtype=float32)>"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "v.scatter_nd_update(indices=[[0, 0], [1, 2]],\n",
    "                    updates=[100., 200.])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "a554f838-fd60-46ad-8ee0-f955139f2277",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=\n",
       "array([[4., 5., 6.],\n",
       "       [1., 2., 3.]], dtype=float32)>"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],\n",
    "                               indices=[1, 0])\n",
    "v.scatter_update(sparse_delta)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9eef3033-a74c-49c7-bc96-5426194e0723",
   "metadata": {},
   "source": [
    "### Tensor Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "107d2660-26c6-4e2e-b2fd-d1aca76446e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "array = tf.TensorArray(dtype=tf.float32, size=3)\n",
    "array = array.write(0, tf.constant([1., 2.]))\n",
    "array = array.write(1, tf.constant([3., 10.]))\n",
    "array = array.write(2, tf.constant([5., 7.]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "f14eaa4e-b45f-4da1-93a5-7a96d6ad4031",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2,), dtype=float32, numpy=array([ 3., 10.], dtype=float32)>"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "array.read(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ed2e84bd-14c7-47ef-bc9d-9d84eb00babb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3, 2), dtype=float32, numpy=\n",
       "array([[1., 2.],\n",
       "       [0., 0.],\n",
       "       [5., 7.]], dtype=float32)>"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "array.stack()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "4692665e-d84d-4889-b721-708ed3349175",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 3.], dtype=float32)>"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean, variance = tf.nn.moments(array.stack(), axes=0)\n",
    "mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "d09a9b4b-617b-44d0-9a38-29a4073e3a1e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(2,), dtype=float32, numpy=array([4.6666665, 8.666667 ], dtype=float32)>"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "variance"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d3629f7-e7d6-4fa2-b47d-dff9f2c32f10",
   "metadata": {},
   "source": [
    "## Custom loss function"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "477cc7e8-238b-4524-89bf-3c2a282d5a9e",
   "metadata": {},
   "source": [
    "Let's start by loading and preparing the California housing dataset. We first load it, then split it into a training set, a validation set and a test set, and finally we scale it:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "8d968ba0-0209-45e0-aa77-edaafe4fc9ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import fetch_california_housing\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "housing = fetch_california_housing()\n",
    "X_train_full, X_test, y_train_full, y_test = train_test_split(\n",
    "    housing.data, housing.target.reshape(-1, 1), random_state=100)\n",
    "X_train, X_valid, y_train, y_valid = train_test_split(\n",
    "    X_train_full, y_train_full, random_state=100)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_valid_scaled = scaler.transform(X_valid)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "427a792d-3fc8-4b23-9752-84db553c492c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def huber_fn(y_true, y_pred):\n",
    "    error = y_true - y_pred\n",
    "    is_small_error = tf.abs(error) < 1\n",
    "    squared_loss = tf.square(error) / 2\n",
    "    linear_loss = tf.abs(error) - 0.5\n",
    "    return tf.where(is_small_error, squared_loss, linear_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "b3f717be-c8a7-45f9-8eb8-5eefc378dd74",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Huber loss')"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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796dbDbXoEGDWLhwIW3atGHkyJE4Ozszffp0ihUrxogRI6wK2sXFhfHjx6c6BECkTdrNetJmtpF2s560mW2k3awnbWYbaTfrZbTNDJqVdQFiY2MZN24c33//Pffv36datWp89NFHtGjRIvE+jRs3TpGkAly/fp23336b3377DbPZTOPGjZkxYwZPPvmkTcELIYQQQojcyeokVQghhBBCiKxm23QrIYQQQgghspAkqUIIIYQQwu5IkiqEEEIIIexOrkxSX3/9dQwGAy+++KLeoditPXv20K5dO/z8/HB1daV48eK0bNmS/fv36x2aXdu+fTv9+vWjYsWKuLu7U758eV577bVUF6oQSlBQEB988AFNmjTBw8MDg8HArl279A7LbsTFxfH+++9TokQJ3Nzc8Pf3Z+vWrXqHZdciIyMZP348LVu2pHDhwhgMBgIDA/UOy679/vvvDBkyhMqVK5M/f35Kly5N165dOX/+vN6h2bVTp07RpUsXypcvj7u7O97e3gQEBLBhwwa9Q8tRPv74YwwGA1WqVLHquFyXpB49epTAwEBcXV31DsWunT9/HgcHB9544w2++uorRo4cya1btwgICGDLli16h2e33n//fXbt2kXHjh2ZPXs23bt356effuLZZ5/l1q1beodnl86dO8dnn33GjRs3qFq1qt7h2J0+ffowffp0XnnlFWbNmoWjoyOtW7dm3759eodmt0JCQpg0aRJnzpyhevXqeoeTI3z22WesXr2apk2bMmvWLAYMGMCePXuoWbMmf//9t97h2a0rV64QERFB7969mTVrFuPGjQOgXbt2yWrGi7Rdv36dTz75hPz581t/sJaLmM1mrV69elq/fv20MmXKaG3atNE7pBwlKipKK1asmNaiRQu9Q7Fbu3fv1kwmU4p9gDZmzBidorJv4eHh2t27dzVN07SVK1dqgLZz5059g7IThw8f1gBt6tSpiftiYmK0J554QqtXr56Okdm32NhYLSgoSNM0Tfv99981QFu8eLG+Qdm5/fv3a3Fxccn2nT9/XnNxcdFeeeUVnaLKmRISErTq1atrlSpV0juUHKFbt27a888/rzVq1EirXLmyVcfmqp7U7777jr///puPP/5Y71ByJHd3d3x8fAgNDdU7FLsVEBCQYg3igIAAChcuzJkzZ3SKyr55eHhQuHBhvcOwS6tWrcLR0ZEB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\n",
      "text/plain": [
       "<Figure size 800x350 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 3.5))\n",
    "z = np.linspace(-4, 4, 200)\n",
    "plt.plot(z, huber_fn(0, z), \"b-\", linewidth=2, label=\"huber($z$)\")\n",
    "plt.plot(z, z**2 / 2, \"b:\", linewidth=1, label=r\"$\\frac{1}{2}z^2$\")\n",
    "plt.plot([-1, -1], [0, huber_fn(0., -1.)], \"r--\")\n",
    "plt.plot([1, 1], [0, huber_fn(0., 1.)], \"r--\")\n",
    "plt.gca().axhline(y=0, color='k')\n",
    "plt.gca().axvline(x=0, color='k')\n",
    "plt.axis([-4, 4, 0, 4])\n",
    "plt.grid(True)\n",
    "plt.xlabel(\"$z$\")\n",
    "plt.legend(fontsize=14)\n",
    "plt.title(\"Huber loss\", fontsize=14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "b7c71101-14a0-44c2-ad08-84abc445810c",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_shape = X_train.shape[1:]\n",
    "\n",
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "8e9bc9cd-33da-488f-b381-d0d379497471",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=huber_fn, optimizer=\"nadam\", metrics=[\"mae\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "2fcbc163-506e-415e-97ea-28f8ee13e031",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.5814 - mae: 0.9394 - val_loss: 0.2259 - val_mae: 0.5281\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.2202 - mae: 0.5150 - val_loss: 0.2049 - val_mae: 0.4980\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bdd42560>"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59c4caa0-2da1-42a1-a9e5-bb025461527e",
   "metadata": {},
   "source": [
    "## Saving/Loading Models with Custom Objects"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "7930be05-0250-400f-92a0-b7f3a90202e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_a_custom_loss.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "ecf5318c-1fa8-4ef7-b854-e381af4bd47d",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_model_with_a_custom_loss.h5\",\n",
    "                                custom_objects={\"huber_fn\": huber_fn})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "4f12fe41-074b-4f61-a52e-8e7e1011147c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.2104 - mae: 0.5007 - val_loss: 0.2003 - val_mae: 0.4895\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.2053 - mae: 0.4924 - val_loss: 0.1950 - val_mae: 0.4825\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bc153490>"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "754d3ed8-5a5e-4848-8b90-bc91ebcabfe3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_huber(threshold=1.0):\n",
    "    def huber_fn(y_true, y_pred):\n",
    "        error = y_true - y_pred\n",
    "        is_small_error = tf.abs(error) < threshold\n",
    "        squared_loss = tf.square(error) / 2\n",
    "        linear_loss = threshold * tf.abs(error) - threshold**2 / 2\n",
    "        return tf.where(is_small_error, squared_loss, linear_loss)\n",
    "    return huber_fn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "10e990b5-d75f-47fe-b5b1-7d23357f0680",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=create_huber(2.0), optimizer=\"nadam\", metrics=[\"mae\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "3f72670c-d0ba-4315-9e5b-f2d79f049434",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.2367 - mae: 0.4922 - val_loss: 0.2199 - val_mae: 0.4847\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.2300 - mae: 0.4866 - val_loss: 0.2158 - val_mae: 0.4797\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e575d6f0>"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "638804bb-5c02-46da-9749-6a5d60407f70",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_a_custom_loss_threshold_2.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "c39376bc-6c0a-4613-8714-85ab1819e138",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_model_with_a_custom_loss_threshold_2.h5\",\n",
    "                                custom_objects={\"huber_fn\": create_huber(2.0)})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "73b2b73a-ca43-4378-a686-48b6a504815f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.2233 - mae: 0.4815 - val_loss: 0.2119 - val_mae: 0.4771\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.2174 - mae: 0.4769 - val_loss: 0.2090 - val_mae: 0.4731\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e56a2350>"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "8d389cb8-8815-4c09-b6c5-9ab2c51850da",
   "metadata": {},
   "outputs": [],
   "source": [
    "class HuberLoss(keras.losses.Loss):\n",
    "    def __init__(self, threshold=1.0, **kwargs):\n",
    "        self.threshold = threshold\n",
    "        super().__init__(**kwargs)\n",
    "    def call(self, y_true, y_pred):\n",
    "        error = y_true - y_pred\n",
    "        is_small_error = tf.abs(error) < self.threshold\n",
    "        squared_loss = tf.square(error) / 2\n",
    "        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2\n",
    "        return tf.where(is_small_error, squared_loss, linear_loss)\n",
    "    def get_config(self):\n",
    "        base_config = super().get_config()\n",
    "        return {**base_config, \"threshold\":self.threshold}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "84e8e2c5-90f6-4772-b949-97d85e30e75f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "4fc3a300-ad3a-4ffa-a2a1-7be326a108ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=HuberLoss(2.), optimizer=\"nadam\", metrics=[\"mae\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "85bc4632-9dd7-4fc9-afe6-1984a0b8bfdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.9270 - mae: 1.0195 - val_loss: 0.2760 - val_mae: 0.5523\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.2632 - mae: 0.5333 - val_loss: 0.2332 - val_mae: 0.5091\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e55ca8c0>"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "f3e29d58-6550-4a5a-b0e0-e4149d312771",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_a_custom_loss_class.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "b2b224b3-aa06-4368-b903-f054e37d1485",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_model_with_a_custom_loss_class.h5\",\n",
    "                                custom_objects={\"HuberLoss\": HuberLoss})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "20de0d72-beaa-4c38-bbe3-b8e02cfa51a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.2449 - mae: 0.5132 - val_loss: 0.2276 - val_mae: 0.5035\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.2361 - mae: 0.5050 - val_loss: 0.2214 - val_mae: 0.4960\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e533f4f0>"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "d6ab8e11-2af9-415a-83fd-0ae7e9d603c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.0"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.loss.threshold"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4851b624-614f-452a-aecc-02e8cfb52723",
   "metadata": {},
   "source": [
    "## Other Custom Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "14c92438-8bb7-4255-b851-9530e2a8d314",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "ac3e5b2e-a8b6-4c6c-a44e-08369cced7b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def my_softplus(z): # return value is just tf.nn.softplus(z)\n",
    "    return tf.math.log(tf.exp(z) + 1.0)\n",
    "\n",
    "def my_glorot_initializer(shape, dtype=tf.float32):\n",
    "    stddev = tf.sqrt(2. / (shape[0] + shape[1]))\n",
    "    return tf.random.normal(shape, stddev=stddev, dtype=dtype)\n",
    "\n",
    "def my_l1_regularizer(weights):\n",
    "    return tf.reduce_sum(tf.abs(0.01 * weights))\n",
    "\n",
    "def my_positive_weights(weights): # return value is just tf.nn.relu(weights)\n",
    "    return tf.where(weights < 0., tf.zeros_like(weights), weights)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "ef29f723-8b84-48b5-9024-07d94283a49d",
   "metadata": {},
   "outputs": [],
   "source": [
    "layer = keras.layers.Dense(1, activation=my_softplus,\n",
    "                           kernel_initializer=my_glorot_initializer,\n",
    "                           kernel_regularizer=my_l1_regularizer,\n",
    "                           kernel_constraint=my_positive_weights)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "afb89d15-daa2-407f-99c8-d699ff1ac66c",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "4e2ff59a-a957-4f79-a185-6db64d26611c",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1, activation=my_softplus,\n",
    "                       kernel_regularizer=my_l1_regularizer,\n",
    "                       kernel_constraint=my_positive_weights,\n",
    "                       kernel_initializer=my_glorot_initializer),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "375e756e-1512-4227-89d6-9083e7d73d5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\", metrics=[\"mae\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "143a1d8a-f74b-48ed-8e2c-136c05419557",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 2.2809 - mae: 1.0610 - val_loss: 1.5601 - val_mae: 0.6346\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.7352 - mae: 0.5668 - val_loss: 0.9136 - val_mae: 0.5351\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e522ff10>"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "5f14019c-dbce-46bd-b7c1-046e807ef387",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_many_custom_parts.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "4f4caef3-28d4-4bba-90e8-9b0b68f063a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\n",
    "    \"my_model_with_many_custom_parts.h5\",\n",
    "    custom_objects={\n",
    "        \"my_l1_regularizer\": my_l1_regularizer,\n",
    "        \"my_positive_weights\": my_positive_weights,\n",
    "        \"my_glorot_initializer\": my_glorot_initializer,\n",
    "        \"my_softplus\": my_softplus,\n",
    "    })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "44bfdd12-6325-493e-9de4-49c76862f036",
   "metadata": {},
   "outputs": [],
   "source": [
    "class MyL1Regularizer(keras.regularizers.Regularizer):\n",
    "    def __init__(self, factor):\n",
    "        self.factor = factor\n",
    "    def __call__(self, weights):\n",
    "        return tf.reduce_sum(tf.abs(self.factor * weights))\n",
    "    def get_config(self):\n",
    "        return {\"factor\": self.factor}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "31fcf4d0-7b0a-47f7-8832-f0f52f666971",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "6d2301eb-0862-4243-8b4a-84a1b9704964",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1, activation=my_softplus,\n",
    "                       kernel_regularizer=MyL1Regularizer(0.01),\n",
    "                       kernel_constraint=my_positive_weights,\n",
    "                       kernel_initializer=my_glorot_initializer),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "90af033b-d153-4fca-906c-038de25d6090",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\", metrics=[\"mae\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "ee0156e6-b39c-453f-b80a-6ff24b2899ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 1.6716 - mae: 0.9038 - val_loss: 0.9233 - val_mae: 0.5598\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.6854 - mae: 0.5393 - val_loss: 0.7119 - val_mae: 0.5216\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e5023c40>"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "0919b652-4176-4d7d-90de-905f5a3880af",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_many_custom_parts.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "276ba04d-a3ae-4223-b74c-bfa047a0e552",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\n",
    "    \"my_model_with_many_custom_parts.h5\",\n",
    "    custom_objects={\n",
    "        \"MyL1Regularizer\": MyL1Regularizer,\n",
    "        \"my_positive_weights\": my_positive_weights,\n",
    "        \"my_glorot_initializer\": my_glorot_initializer,\n",
    "        \"my_softplus\": my_softplus,\n",
    "    })"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f614699-dc72-4717-b3fb-71665c89e293",
   "metadata": {},
   "source": [
    "## Custom Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "b41020ee-ca81-4a21-82cd-28f99f4470ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "6da7fbad-c6fc-43c6-98d0-8f635f0dbe5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "f6babba6-998a-4487-8f36-cbb2f06816ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\", metrics=[create_huber(2.0)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "c8baa7ae-a6e1-4409-9633-882f456fd6ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 2ms/step - loss: 1.9466 - huber_fn: 0.8516\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.5610 - huber_fn: 0.2651\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87e50d3910>"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25ce1865-e5ec-4b5b-a3f3-f0b00c1b0be8",
   "metadata": {},
   "source": [
    "**Note**: if you use the same function as the loss and a metric, you may be surprised to see different results. This is generally just due to floating point precision errors: even though the mathematical equations are equivalent, the operations are not run in the same order, which can lead to small differences. Moreover, when using sample weights, there's more than just precision errors:\n",
    "* the loss since the start of the epoch is the mean of all batch losses seen so far. Each batch loss is the sum of the weighted instance losses divided by the _batch size_ (not the sum of weights, so the batch loss is _not_ the weighted mean of the losses).\n",
    "* the metric since the start of the epoch is equal to the sum of weighted instance losses divided by sum of all weights seen so far. In other words, it is the weighted mean of all the instance losses. Not the same thing.\n",
    "\n",
    "If you do the math, you will find that loss = metric * mean of sample weights (plus some floating point precision error)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "ce29ec14-c478-44c0-8445-9d6b98c3871f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=create_huber(2.0), optimizer=\"nadam\", metrics=[create_huber(2.0)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "981feb78-16c9-4b63-9373-5e3257a7d997",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.1269 - huber_fn: 0.2505\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 4ms/step - loss: 0.1215 - huber_fn: 0.2393\n"
     ]
    }
   ],
   "source": [
    "sample_weight = np.random.rand(len(y_train))\n",
    "history = model.fit(X_train_scaled, y_train, epochs=2, sample_weight=sample_weight)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "fb3d4050-f0ee-48ab-ab54-42c463c7d338",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.12692150473594666, 0.12477653969289013)"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history.history[\"loss\"][0], history.history[\"huber_fn\"][0] * sample_weight.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3dd8b77f-04f6-4ecc-a11e-b837a3afb42d",
   "metadata": {},
   "source": [
    "### Streaming metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "8e61acb1-a029-4bbe-bc52-7e2b748e5741",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=0.8>"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision = keras.metrics.Precision()\n",
    "precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "fded8d65-5c67-4693-8f7b-291449a460f5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=0.5>"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "5da271da-84d6-4a58-9ee8-4255f1bfb8ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=0.5>"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision.result()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "8e710343-10cf-46db-9039-1ceb79289597",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Variable 'true_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>,\n",
       " <tf.Variable 'false_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>]"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision.variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "ba8522bd-ec8d-40ac-bb0c-2e6c9d4a9c38",
   "metadata": {},
   "outputs": [],
   "source": [
    "precision.reset_states()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8a15cd3-d208-4a13-b148-974fefbe9d7a",
   "metadata": {},
   "source": [
    "Creating a streaming metric:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "2d3bc4ad-bf0c-4e18-9ebb-742edaf2edab",
   "metadata": {},
   "outputs": [],
   "source": [
    "class HuberMetric(keras.metrics.Metric):\n",
    "    def __init__(self, threshold=1.0, **kwargs):\n",
    "        super().__init__(**kwargs) # hadles base args\n",
    "        self.threshold = threshold\n",
    "        self.huber_fn = create_huber(threshold)\n",
    "        self.total = self.add_weight(\"total\", initializer=\"zeros\")\n",
    "        self.count = self.add_weight(\"count\", initializer=\"zeros\")\n",
    "    def update_state(self, y_true, y_pred, sample_weight=None):\n",
    "        metric = self.huber_fn(y_true, y_pred)\n",
    "        self.total.assign_add(tf.reduce_sum(metric))\n",
    "        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))\n",
    "    def result(self):\n",
    "        return self.total / self.count\n",
    "    def get_config(self):\n",
    "        base_config = super().get_config()\n",
    "        return {**base_config, \"threshold\":self.threshold}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "98c69a3d-f958-40a4-ab98-de3da3900ac6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=14.0>"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m = HuberMetric(2.)\n",
    "\n",
    "# total = 2 * |10 - 2| - 2²/2 = 14\n",
    "# count = 1\n",
    "# result = 14 / 1 = 14\n",
    "m(tf.constant([[2.]]), tf.constant([[10.]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "60b7c71e-86bf-4408-bd4e-e60ae45538b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=7.0>"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# total = total + (|1 - 0|² / 2) + (2 * |9.25 - 5| - 2² / 2) = 14 + 7 = 21\n",
    "# count = count + 2 = 3\n",
    "# result = total / count = 21 / 3 = 7\n",
    "m(tf.constant([[0.], [5.]]), tf.constant([[1.], [9.25]]))\n",
    "\n",
    "m.result()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "aa903381-d254-4c26-845c-d9d63367df34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Variable 'total:0' shape=() dtype=float32, numpy=21.0>,\n",
       " <tf.Variable 'count:0' shape=() dtype=float32, numpy=3.0>]"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m.variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "707c9271-d14d-4582-a655-7ef5f11b74c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Variable 'total:0' shape=() dtype=float32, numpy=0.0>,\n",
       " <tf.Variable 'count:0' shape=() dtype=float32, numpy=0.0>]"
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m.reset_states()\n",
    "m.variables"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e1da540-13f3-474a-a8db-8441c2bb6d6c",
   "metadata": {},
   "source": [
    "Let's check that the `HuberMetric` class works well:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "a9b9289b-6ed3-4969-8486-556a686df4a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "09a322f4-3c2d-46c7-a80c-aba46c5130d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "ae644fa4-fc2e-4935-a302-5115dede84cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=create_huber(2.0), optimizer=\"nadam\", metrics=[HuberMetric(2.0)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "146beda5-2a42-43e9-96bf-00481cc0e913",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.7916 - huber_metric: 0.7916\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.2805 - huber_metric: 0.2805\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bdba0850>"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "7adf040b-39aa-47c7-a0ce-3762458810bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_a_custom_metric.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "4347a047-9a4b-4681-af95-8f79ab6238a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_model_with_a_custom_metric.h5\",\n",
    "                                custom_objects={\"huber_fn\": create_huber(2.0),\n",
    "                                                \"HuberMetric\": HuberMetric})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "d103c864-2925-4ab0-ad54-fd5917009c1e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.2643 - huber_metric: 0.2643\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.2533 - huber_metric: 0.2533\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bdadf940>"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0608c07-30a6-4ca7-b9dc-b272a3f798c5",
   "metadata": {},
   "source": [
    "**Warning**: In TF 2.2, tf.keras adds an extra first metric in `model.metrics` at position 0 (see [TF issue #38150](https://github.com/tensorflow/tensorflow/issues/38150)). This forces us to use `model.metrics[-1]` rather than `model.metrics[0]` to access the `HuberMetric`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "5e1c6372-e9b2-4573-98c6-c739c0ecd1cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.0"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.metrics[-1].threshold"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87718a23-ac53-4dee-8fb7-ac30f2542907",
   "metadata": {},
   "source": [
    "Looks like it works fine! More simply, we could have created the class like this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "916894de-ecc6-4fa7-80ad-9ca44bd379bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "class HuberMetric(keras.metrics.Mean):\n",
    "    def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):\n",
    "        self.threshold = threshold\n",
    "        self.huber_fn = create_huber(threshold)\n",
    "        super().__init__(name=name, dtype=dtype)\n",
    "    def update_state(self, y_true, y_pred, sample_weight=None):\n",
    "        metric = self.huber_fn(y_true, y_pred)\n",
    "        super(HuberMetric, self).update_state(metric, sample_weight)\n",
    "    def get_config(self):\n",
    "        base_config = super().get_config()\n",
    "        return {**base_config, \"threshold\": self.threshold}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a9fdd85-ca54-4dec-8285-13107858f968",
   "metadata": {},
   "source": [
    "This class handles shapes better, and it also supports sample weights."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "614057c9-0568-455d-8cdb-cda863d92f1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "13532e6d-1d6d-47c7-9954-d227304143b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"selu\", kernel_initializer=\"lecun_normal\",\n",
    "                       input_shape=input_shape),\n",
    "    keras.layers.Dense(1),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "7c3018f3-5129-4362-85d4-5c97f88fa70b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=keras.losses.Huber(2.0), optimizer=\"nadam\", weighted_metrics=[HuberMetric(2.0)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "2a3d7a5c-861d-4e44-97b9-73210fed2180",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 0.5217 - HuberMetric: 1.0474\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.1391 - HuberMetric: 0.2792\n"
     ]
    }
   ],
   "source": [
    "sample_weight = np.random.rand(len(y_train))\n",
    "history = model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32),\n",
    "                    epochs=2, sample_weight=sample_weight)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "172c9e18-2bb0-4127-9001-d827e89f4385",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.5217181444168091, 0.5217185791654395)"
      ]
     },
     "execution_count": 132,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history.history[\"loss\"][0], history.history[\"HuberMetric\"][0] * sample_weight.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "8f3d4d28-4074-4d9c-bbc5-af710a7bd7bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_a_custom_metric_v2.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "id": "17cf7576-301d-4d10-aacd-f37b9185e083",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_model_with_a_custom_metric_v2.h5\",\n",
    "                                custom_objects={\"HuberMetric\": HuberMetric})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "13daf22d-8462-47f1-8558-5b0f64ed4cc1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.2501 - HuberMetric: 0.2501\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.2359 - HuberMetric: 0.2359\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bd7567a0>"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "c594de66-eb85-45cf-be53-074197847b99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.0"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.metrics[-1].threshold"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa94ca6e-31ad-455e-afcb-b595440d0b34",
   "metadata": {},
   "source": [
    "## Custom Layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "id": "d7eaf39d-de06-4ab3-9114-e7a11e4d6f65",
   "metadata": {},
   "outputs": [],
   "source": [
    "exponential_layer = keras.layers.Lambda(lambda X: tf.exp(X))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "id": "a9386601-67ce-40a2-84bc-e36cb722a9ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.36787945, 1.        , 2.7182817 ], dtype=float32)>"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "exponential_layer([-1., 0., 1.])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8044039a-edcc-4fd1-8c01-23ccb1300aa2",
   "metadata": {},
   "source": [
    "Adding an exponential layer at the output of a regression model can be useful if the values to predict are positive and with very different scales (e.g., 0.001, 10., 10000):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "id": "6fa2f45c-80b9-433a-af59-3e7a233971d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "id": "c4af27fb-a605-4ed3-89f0-25e6d92b572f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.8854 - val_loss: 0.5288\n",
      "Epoch 2/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.5248 - val_loss: 0.4673\n",
      "Epoch 3/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.4706 - val_loss: 0.4462\n",
      "Epoch 4/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.4475 - val_loss: 0.4344\n",
      "Epoch 5/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.4343 - val_loss: 0.4153\n",
      "162/162 [==============================] - 0s 2ms/step - loss: 0.3986\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.39861032366752625"
      ]
     },
     "execution_count": 140,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"relu\", input_shape=input_shape),\n",
    "    keras.layers.Dense(1),\n",
    "    exponential_layer\n",
    "])\n",
    "model.compile(loss=\"mse\", optimizer=\"sgd\")\n",
    "model.fit(X_train_scaled, y_train, epochs=5,\n",
    "          validation_data=(X_valid_scaled, y_valid))\n",
    "model.evaluate(X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "8ccef533-28d3-4ebd-a715-d9c976ddaec6",
   "metadata": {},
   "outputs": [],
   "source": [
    "class MyDense(keras.layers.Layer):\n",
    "    def __init__(self, units, activation=None, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units = units\n",
    "        self.activation = keras.activations.get(activation)\n",
    "        \n",
    "    def build(self, batch_input_shape):\n",
    "        self.kernel = self.add_weight(\n",
    "            name=\"kernel\", shape=[batch_input_shape[-1], self.units],\n",
    "            initializer=\"glorot_normal\")\n",
    "        self.bias = self.add_weight(\n",
    "            name = \"bias\", shape=[self.units], initializer=\"zeros\")\n",
    "        super().build(batch_input_shape) # must be at the end\n",
    "        \n",
    "    def call(self, X):\n",
    "        return self.activation(X @ self.kernel + self.bias)\n",
    "    \n",
    "    def compute_output_shape(self, batch_input_shape):\n",
    "        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])\n",
    "    \n",
    "    def get_config(self):\n",
    "        base_config = super().get_config()\n",
    "        return {**base_config, \"units\": self.units,\n",
    "                \"activation\": keras.activations.serialize(self.activation)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "f6bd9879-de3a-488d-ba23-6557fd3bd3bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "c07fa486-68bb-4d04-a1e0-0a0ba0082f58",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    MyDense(30, activation=\"relu\", input_shape=input_shape),\n",
    "    MyDense(1)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "61f82b20-a952-4367-8c75-b1b53f829a8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 1.8249 - val_loss: 0.7362\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.6273 - val_loss: 0.5366\n",
      "162/162 [==============================] - 0s 2ms/step - loss: 0.5058\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.5058388710021973"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\")\n",
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))\n",
    "model.evaluate(X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "fde27be4-e695-4262-ba22-efb4ef82a3a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"my_model_with_a_custom_layer.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "4f61d7b5-d3da-4ebe-bd30-5371deb26a52",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_model_with_a_custom_layer.h5\",\n",
    "                                custom_objects={\"MyDense\": MyDense})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "adc928b4-8e5d-4220-9710-0b8994696b62",
   "metadata": {},
   "outputs": [],
   "source": [
    "class MyMultiLayer(keras.layers.Layer):\n",
    "    def call(self, X):\n",
    "        X1, X2 = X\n",
    "        print(\"X1.shape: \", X1.shape ,\" X2.shape: \", X2.shape) # Debugging of custom layer\n",
    "        return X1 + X2, X1 * X2\n",
    "    \n",
    "    def compute_output_shape(self, batch_input_shape):\n",
    "        batch_input_shape1, batch_input_shape2 = batch_input_shape\n",
    "        return [batch_input_shape1, batch_input_shape2]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c28c964c-d8d9-4153-bf4a-1c818cb1683a",
   "metadata": {},
   "source": [
    "Our custom layer can be called using the functional API like this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "9c6e15ce-0902-4d12-b25a-a0b3b9347587",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X1.shape:  (None, 2)  X2.shape:  (None, 2)\n"
     ]
    }
   ],
   "source": [
    "inputs1 = keras.layers.Input(shape=[2])\n",
    "inputs2 = keras.layers.Input(shape=[2])\n",
    "outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a960aa3-ea7d-482d-9780-772c60b7abb9",
   "metadata": {},
   "source": [
    "Note that the `call()` method receives symbolic inputs, whose shape is only partially specified (at this stage, we don't know the batch size, which is why the first dimension is `None`):\n",
    "\n",
    "We can also pass actual data to the custom layer. To test this, let's split each dataset's inputs into two parts, with four features each:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "7ec85692-3b69-496b-88a1-a98d252dfaa7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((11610, 8), (11610, 4))"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def split_data(data):\n",
    "    columns_count = data.shape[-1]\n",
    "    half = columns_count // 2\n",
    "    return data[:, :half], data[:, half:]\n",
    "\n",
    "X_train_scaled_A, X_train_scaled_B = split_data(X_train_scaled)\n",
    "X_valid_scaled_A, X_valid_scaled_B = split_data(X_valid_scaled)\n",
    "X_test_scaled_A, X_test_scaled_B = split_data(X_test_scaled)\n",
    "\n",
    "X_train_scaled.shape, X_train_scaled_B.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9dda4a9a-1f4b-4972-9f56-46fbe28dd381",
   "metadata": {},
   "source": [
    "Now notice that the shapes are fully specified:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "0a5ee8a7-cc58-4dff-9512-c1de1f3d8fc2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X1.shape:  (11610, 4)  X2.shape:  (11610, 4)\n"
     ]
    }
   ],
   "source": [
    "outputs1, outputs2 = MyMultiLayer()((X_train_scaled_A, X_train_scaled_B))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07c6467b-087b-4665-bba9-afbc643c02de",
   "metadata": {},
   "source": [
    "Let's build a more complete model using the functional API (this is just a toy example, don't expect awesome performance):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "c4ccdbd2-5f66-4c3e-9de7-6532e59ccf3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X1.shape:  (None, 4)  X2.shape:  (None, 4)\n"
     ]
    }
   ],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)\n",
    "\n",
    "input_A = keras.layers.Input(shape=X_train_scaled_A.shape[-1])\n",
    "input_B = keras.layers.Input(shape=X_train_scaled_B.shape[-1])\n",
    "hidden_A, hidden_B = MyMultiLayer()((input_A, input_B))\n",
    "hidden_A = keras.layers.Dense(30, activation=\"selu\")(hidden_A)\n",
    "hidden_B = keras.layers.Dense(30, activation=\"selu\")(hidden_B)\n",
    "concat = keras.layers.Concatenate()((hidden_A, hidden_B))\n",
    "output = keras.layers.Dense(1)(concat)\n",
    "model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "46f654a3-f2de-4fab-8860-7a2194cbaffd",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "552f5baf-74b8-4264-a6f8-f73a8ad1f2f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "X1.shape:  (None, 4)  X2.shape:  (None, 4)\n",
      "X1.shape:  (None, 4)  X2.shape:  (None, 4)\n",
      "336/363 [==========================>...] - ETA: 0s - loss: 2.0456X1.shape:  (None, 4)  X2.shape:  (None, 4)\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 1.9632 - val_loss: 0.9559\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.9453 - val_loss: 0.9390\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bd457430>"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit((X_train_scaled_A, X_train_scaled_B), y_train, epochs=2,\n",
    "          validation_data=((X_valid_scaled_A, X_valid_scaled_B), y_valid))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ebf4899-46c5-46cc-9331-b838239d194b",
   "metadata": {},
   "source": [
    "Now let's create a layer with a different behavior during training and testing:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "502b07dd-a2f5-4453-a2af-7832f419b043",
   "metadata": {},
   "outputs": [],
   "source": [
    "class AddGaussianNoise(keras.layers.Layer):\n",
    "    def __init__(self, stddev, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.stddev = stddev\n",
    "        \n",
    "    def call(self, X, training=None):\n",
    "        if training:\n",
    "            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)\n",
    "            return X + noise\n",
    "        else:\n",
    "            return X\n",
    "    \n",
    "    def compute_output_shape(self, batch_input_shape):\n",
    "        return batch_input_shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37c25d8e-0ffc-4f1c-9c89-7efce8171a78",
   "metadata": {},
   "source": [
    "Here's a simple model that uses this custom layer:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "3c124cfa-622d-48b0-a36e-3bb3163f4148",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)\n",
    "\n",
    "model = keras.models.Sequential([\n",
    "    AddGaussianNoise(stddev=1.0),\n",
    "    keras.layers.Dense(30, activation=\"selu\"),\n",
    "    keras.layers.Dense(1)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "29b4dc60-5d73-4d27-9d73-3113666cf118",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 2s 3ms/step - loss: 2.2761 - val_loss: 0.8232\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 1.0268 - val_loss: 0.7894\n",
      "162/162 [==============================] - 0s 1ms/step - loss: 0.7898\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.789842426776886"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\")\n",
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))\n",
    "model.evaluate(X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "60760ded-5eec-4618-8c18-e7a6b7cc7238",
   "metadata": {},
   "source": [
    "## Custom Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "d2e387df-4a14-4fe0-9cde-17859f663c85",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_new_scaled = X_test_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "e5a10bb7-650b-4769-bdb4-7811cb80a6cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "class ResidualBlock(keras.layers.Layer):\n",
    "    def __init__(self, n_layers, n_neurons, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.hidden = [keras.layers.Dense(n_neurons, activation=\"elu\",\n",
    "                                          kernel_initializer=\"he_normal\")\n",
    "                       for _ in range(n_layers)]\n",
    "\n",
    "    def call(self, inputs):\n",
    "        Z = inputs\n",
    "        for layer in self.hidden:\n",
    "            Z = layer(Z)\n",
    "        return inputs + Z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "e9640995-60da-41ad-a8fd-54311470fb73",
   "metadata": {},
   "outputs": [],
   "source": [
    "class ResidualRegressor(keras.models.Model):\n",
    "    def __init__(self, output_dim, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.hidden1 = keras.layers.Dense(30, activation=\"elu\",\n",
    "                                          kernel_initializer=\"he_normal\")\n",
    "        self.block1 = ResidualBlock(2, 30)\n",
    "        self.block2 = ResidualBlock(2, 30)\n",
    "        self.out = keras.layers.Dense(output_dim)\n",
    "\n",
    "    def call(self, inputs):\n",
    "        Z = self.hidden1(inputs)\n",
    "        for _ in range(1 + 3):\n",
    "            Z = self.block1(Z)\n",
    "        Z = self.block2(Z)\n",
    "        return self.out(Z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "3a651c00-0b86-415b-8712-0f7692bcb2c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "6270f65a-d5a1-489b-84cf-946af6a27668",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "363/363 [==============================] - 3s 3ms/step - loss: 13.0209\n",
      "Epoch 2/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 1.0925\n",
      "Epoch 3/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 2.5181\n",
      "Epoch 4/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 1.0064\n",
      "Epoch 5/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 2.2745\n",
      "162/162 [==============================] - 0s 2ms/step - loss: 0.7320\n",
      "162/162 [==============================] - 0s 2ms/step\n"
     ]
    }
   ],
   "source": [
    "model = ResidualRegressor(1)\n",
    "model.compile(loss=\"mse\", optimizer=\"nadam\")\n",
    "history = model.fit(X_train_scaled, y_train, epochs=5)\n",
    "score = model.evaluate(X_test_scaled, y_test)\n",
    "y_pred = model.predict(X_new_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "349eb4cb-6c90-4e7e-8a3c-707ee25367ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as dense_1_layer_call_fn, dense_1_layer_call_and_return_conditional_losses, dense_2_layer_call_fn, dense_2_layer_call_and_return_conditional_losses, dense_3_layer_call_fn while saving (showing 5 of 8). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: my_custom_model.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: my_custom_model.ckpt/assets\n"
     ]
    }
   ],
   "source": [
    "model.save(\"my_custom_model.ckpt\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "d6c4cade-bd65-435e-81e2-2008990f1943",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.load_model(\"my_custom_model.ckpt\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "cbdfb07f-fd5c-4560-af4a-ad2cc48f4a96",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "363/363 [==============================] - 3s 3ms/step - loss: 1.2029\n",
      "Epoch 2/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.5793\n",
      "Epoch 3/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 1.7409\n",
      "Epoch 4/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 1.1450\n",
      "Epoch 5/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 2.1534\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(X_train_scaled, y_train, epochs=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a031e260-ec81-4b11-8954-4b06194c2211",
   "metadata": {},
   "source": [
    "We could have defined the model using the sequential API instead:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "8ee35094-2073-40b9-94d7-e96a99ec252c",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "4d5a4a94-f38f-43e5-94af-5008e2a4654b",
   "metadata": {},
   "outputs": [],
   "source": [
    "block1 = ResidualBlock(2, 30)\n",
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"elu\", kernel_initializer=\"he_normal\"),\n",
    "    block1, block1, block1, block1,\n",
    "    ResidualBlock(2, 30),\n",
    "    keras.layers.Dense(1)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "3a747dd3-76f1-4429-8c3a-2347bff01a63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "363/363 [==============================] - 2s 2ms/step - loss: 1.3088\n",
      "Epoch 2/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.4745\n",
      "Epoch 3/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 1.1518\n",
      "Epoch 4/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.4506\n",
      "Epoch 5/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 0.9672\n",
      "162/162 [==============================] - 0s 2ms/step - loss: 0.4005\n",
      "162/162 [==============================] - 0s 2ms/step\n"
     ]
    }
   ],
   "source": [
    "model.compile(loss=\"mse\", optimizer=\"nadam\")\n",
    "history = model.fit(X_train_scaled, y_train, epochs=5)\n",
    "score = model.evaluate(X_test_scaled, y_test)\n",
    "y_pred = model.predict(X_new_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3fca25eb-13ab-4cf1-a4f1-52dae320e9d1",
   "metadata": {},
   "source": [
    "## Losses and Metrics Based on Model Internals"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cd13362-a2a3-454a-8684-9f6fcb3e853e",
   "metadata": {},
   "source": [
    "**Note**: the following code has two differences with the code in the book:\n",
    "1. It creates a `keras.metrics.Mean()` metric in the constructor and uses it in the `call()` method to track the mean reconstruction loss. Since we only want to do this during training, we add a `training` argument to the `call()` method, and if `training` is `True`, then we update `reconstruction_mean` and we call `self.add_metric()` to ensure it's displayed properly.\n",
    "2. Due to an issue introduced in TF 2.2 ([#46858](https://github.com/tensorflow/tensorflow/issues/46858)), we must not call `super().build()` inside the `build()` method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "749da3f3-e4be-47d7-b21f-68f7dc8b24ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "class ReconstructingRegressor(keras.Model):\n",
    "    def __init__(self, output_dim, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.hidden = [keras.layers.Dense(30, activation=\"selu\",\n",
    "                                          kernel_initializer=\"lecun_normal\")\n",
    "                       for _ in range(5)]\n",
    "        self.out = keras.layers.Dense(output_dim)\n",
    "        self.reconstruction_mean = keras.metrics.Mean(name=\"reconstruction_error\")\n",
    "\n",
    "    def build(self, batch_input_shape):\n",
    "        n_inputs = batch_input_shape[-1]\n",
    "        self.reconstruct = keras.layers.Dense(n_inputs)\n",
    "        #super().build(batch_input_shape)\n",
    "\n",
    "    def call(self, inputs, training=None):\n",
    "        Z = inputs\n",
    "        for layer in self.hidden:\n",
    "            Z = layer(Z)\n",
    "        reconstruction = self.reconstruct(Z)\n",
    "        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))\n",
    "        self.add_loss(0.05 * recon_loss)\n",
    "        if training:\n",
    "            result = self.reconstruction_mean(recon_loss)\n",
    "            self.add_metric(result)\n",
    "        return self.out(Z)        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "7ab8dd2e-bcba-4d23-aeb3-152ac215d533",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "4149a2f6-951b-4671-a49a-4d74a4982d96",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "363/363 [==============================] - 13s 4ms/step - loss: 0.7587 - reconstruction_error: 0.8310\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 2s 5ms/step - loss: 0.4743 - reconstruction_error: 0.3981\n",
      "162/162 [==============================] - 1s 4ms/step\n"
     ]
    }
   ],
   "source": [
    "model = ReconstructingRegressor(1)\n",
    "model.compile(loss=\"mse\", optimizer=\"nadam\")\n",
    "history = model.fit(X_train_scaled, y_train, epochs=2)\n",
    "y_pred = model.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6524353-72ab-42aa-83db-b1865ce551ba",
   "metadata": {},
   "source": [
    "## Computing Gradients with Autodiff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "63887ceb-5b4a-46bf-a231-4188922bb946",
   "metadata": {},
   "outputs": [],
   "source": [
    "def f(w1, w2):\n",
    "    return 3 * w1 ** 2 + 2 * w1 * w2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "24b39aaf-3c5f-4490-bbe4-e0f5ca0c347a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "36.000003007075065"
      ]
     },
     "execution_count": 172,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "w1, w2 = 5, 3\n",
    "eps = 1e-6\n",
    "(f(w1 + eps, w2) - f(w1, w2)) / eps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "f7c10d7a-8df3-403b-a4bb-1d169d3a48ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10.000000003174137"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(f(w1, w2 + eps) - f(w1, w2)) / eps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "c4ddda05-660a-464d-85ed-cf7f1e34a278",
   "metadata": {},
   "outputs": [],
   "source": [
    "w1, w2 = tf.Variable(5.), tf.Variable(3.)\n",
    "with tf.GradientTape() as tape:\n",
    "    z = f(w1, w2)\n",
    "    \n",
    "gradients = tape.gradient(z, [w1, w2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "887ecc8f-e4ff-4073-b09e-fed69f41e653",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,\n",
       " <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]"
      ]
     },
     "execution_count": 175,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gradients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "227fd8b0-ea99-48f9-9b68-70c8e546d4b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)\n"
     ]
    }
   ],
   "source": [
    "with tf.GradientTape() as tape:\n",
    "    z = f(w1, w2)\n",
    "    \n",
    "dz_dw1 = tape.gradient(z, w1)\n",
    "try:\n",
    "    dz_dw2 = tape.gradient(z, w2)\n",
    "except RuntimeError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "49643c33-be15-4332-b731-c161b659d4e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "with tf.GradientTape(persistent=True) as tape:\n",
    "    z = f(w1, w2)\n",
    "    \n",
    "dz_dw1 = tape.gradient(z, w1)\n",
    "dz_dw2 = tape.gradient(z, w2) # works now!\n",
    "del tape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "98ed99ab-05e1-40cc-ac66-c715efc00d65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,\n",
       " <tf.Tensor: shape=(), dtype=float32, numpy=10.0>)"
      ]
     },
     "execution_count": 178,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dz_dw1, dz_dw2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "f77dcc6d-4f5b-42b7-aa0a-7333af32f1f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "c1, c2 = tf.constant(5.), tf.constant(3.)\n",
    "with tf.GradientTape() as tape:\n",
    "    z = f(c1, c2)\n",
    "\n",
    "gradients = tape.gradient(z, [c1, c2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "6eda8579-d36a-481d-973e-2c3753890d83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[None, None]"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gradients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "id": "cfc501ef-66b6-4288-82d3-581c4592de74",
   "metadata": {},
   "outputs": [],
   "source": [
    "with tf.GradientTape() as tape:\n",
    "    tape.watch(c1)\n",
    "    tape.watch(c2)\n",
    "    z = f(c1, c2)\n",
    "    \n",
    "gradients = tape.gradient(z, [c1, c2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "id": "a5b08f51-0e23-4fe8-85e5-919b79c66923",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,\n",
       " <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]"
      ]
     },
     "execution_count": 182,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gradients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "id": "951042b1-fe3d-4d18-8c03-a1fb697381c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=136.0>,\n",
       " <tf.Tensor: shape=(), dtype=float32, numpy=30.0>]"
      ]
     },
     "execution_count": 183,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "with tf.GradientTape() as tape:\n",
    "    z1 = f(w1, w2 + 2.)\n",
    "    z2 = f(w1, w2 + 5.)\n",
    "    z3 = f(w1, w2 + 7.)\n",
    "    \n",
    "tape.gradient([z1, z2, z3], [w1, w2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "fcbf784d-a681-4dff-9e02-d939fbc149ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "with tf.GradientTape(persistent=True) as tape:\n",
    "    z1 = f(w1, w2 + 2.)\n",
    "    z2 = f(w1, w2 + 5.)\n",
    "    z3 = f(w1, w2 + 7.)\n",
    "    \n",
    "tf.reduce_sum(tf.stack([tape.gradient(z, [w1, w2]) for z in (z1, z2, z3)]), axis=0)\n",
    "del tape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "252497f7-562e-40cb-b404-231fa7a66589",
   "metadata": {},
   "outputs": [],
   "source": [
    "with tf.GradientTape(persistent=True) as hessian_tape:\n",
    "    with tf.GradientTape() as jacobian_tape:\n",
    "        z = f(w1, w2)\n",
    "    jacobians = jacobian_tape.gradient(z, [w1, w2])\n",
    "hessians = [hessian_tape.gradient(jacobian, [w1, w2])\n",
    "            for jacobian in jacobians]\n",
    "del hessian_tape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "id": "b98e4b40-2c83-4481-bccd-db4149e80a46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,\n",
       " <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]"
      ]
     },
     "execution_count": 186,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "jacobians"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "1c1951c7-4e25-42d2-9e51-185c34b7f995",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[<tf.Tensor: shape=(), dtype=float32, numpy=6.0>,\n",
       "  <tf.Tensor: shape=(), dtype=float32, numpy=2.0>],\n",
       " [<tf.Tensor: shape=(), dtype=float32, numpy=2.0>, None]]"
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hessians"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "e8667480-643d-447b-ba19-e75ef33afe4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=30.0>, None]"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def f(w1, w2):\n",
    "    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)\n",
    "\n",
    "with tf.GradientTape() as tape:\n",
    "    z = f(w1, w2)\n",
    "    \n",
    "tape.gradient(z, [w1, w2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "id": "6e0ddfb9-e050-4f79-8a2f-a31a899acbfa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=nan>]"
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = tf.Variable(100.)\n",
    "with tf.GradientTape() as tape:\n",
    "    z = my_softplus(x)\n",
    "    \n",
    "tape.gradient(z, [x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "id": "07532e0f-5b26-46f7-8222-135da428db79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=30.0>"
      ]
     },
     "execution_count": 190,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.math.log(tf.exp(tf.constant(30., dtype=tf.float32)) + 1.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "id": "47713ba9-2cd1-4e6f-b106-6c3d01a059f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor: shape=(), dtype=float32, numpy=nan>]"
      ]
     },
     "execution_count": 191,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z = tf.Variable([100.])\n",
    "with tf.GradientTape() as tape:\n",
    "    z = my_softplus(x)\n",
    "    \n",
    "tape.gradient(z, [x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "id": "abc6dca6-446d-4f8c-b3d1-f470bd9d2bc3",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.custom_gradient\n",
    "def my_better_softplus(z):\n",
    "    exp = tf.exp(z)\n",
    "    def my_softplus_gradients(grad):\n",
    "        return grad / (1 + 1 / exp)\n",
    "    return tf.math.log(exp + 1), my_softplus_gradients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "8d9b7d2e-c5d3-44d3-8e03-d92ae2894709",
   "metadata": {},
   "outputs": [],
   "source": [
    "def my_better_softplus(z):\n",
    "    return tf.where(z > 30., z, tf.math.log(tf.exp(z) + 1.))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "5f53704a-eb03-4439-83ae-751842d99c88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(<tf.Tensor: shape=(1,), dtype=float32, numpy=array([1000.], dtype=float32)>,\n",
       " [<tf.Tensor: shape=(1,), dtype=float32, numpy=array([nan], dtype=float32)>])"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = tf.Variable([1000.])\n",
    "with tf.GradientTape() as tape:\n",
    "    z = my_better_softplus(x)\n",
    "\n",
    "z, tape.gradient(z, [x])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f6e92578-674f-474a-8fdd-7a40dc3cdace",
   "metadata": {},
   "source": [
    "# Custom Training Loops"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "10d9f499-587c-47ea-b273-4cae65567675",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "58314c48-da05-4b84-abdb-ad9cdcbb0fe5",
   "metadata": {},
   "outputs": [],
   "source": [
    "l2_reg = keras.regularizers.l2(0.05)\n",
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(30, activation=\"elu\", kernel_initializer=\"he_normal\",\n",
    "                       kernel_regularizer=l2_reg),\n",
    "    keras.layers.Dense(1, kernel_regularizer=l2_reg)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "id": "81c25329-b471-4b24-bf51-529c559e70e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def random_batch(X, y, batch_size=32):\n",
    "    idx = np.random.randint(len(X), size=batch_size)\n",
    "    return X[idx], y[idx]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "id": "a6c0214f-0e65-4f30-b1d0-370c910e9fd4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_status_bar(iteration, total, loss, metrics=None):\n",
    "    metrics = \" - \".join([\"{}: {:.4f}\".format(m.name, m.result())\n",
    "                         for m in [loss] + (metrics or [])])\n",
    "    end = \"\" if iteration < total else \"\\n\"\n",
    "    print(\"\\r{}/{} - \".format(iteration, total) + metrics,\n",
    "          end=end)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "id": "eceb7dc2-7d03-4002-b405-cc7958fd3420",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "50/50 - loss: 0.0900 - mean_square: 858.5000\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "\n",
    "mean_loss = keras.metrics.Mean(name=\"loss\")\n",
    "mean_square = keras.metrics.Mean(name=\"mean_square\")\n",
    "for i in range(1, 50 + 1):\n",
    "    loss = 1 / i\n",
    "    mean_loss(loss)\n",
    "    mean_square(i**2)\n",
    "    print_status_bar(i, 50, mean_loss, [mean_square])\n",
    "    time.sleep(0.05)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80eaa445-6538-46ec-a3a6-031368b966f4",
   "metadata": {},
   "source": [
    "A fancier version with a progress bar:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "id": "33f52d2a-2614-40e5-8757-49556d1a9da4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def progress_bar(iteration, total, size=30):\n",
    "    running = iteration < total\n",
    "    c = \">\" if running else \"=\"\n",
    "    p = (size - 1) * iteration // total\n",
    "    fmt = \"{{:-{}d}}/{{}} [{{}}]\".format(len(str(total)))\n",
    "    params = [iteration, total, \"=\" * p + c + \".\" * (size - p - 1)]\n",
    "    return fmt.format(*params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "2bea3adf-f75c-4e87-84a6-ee0fbbe3642c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "' 3500/10000 [=>....]'"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "progress_bar(3500, 10000, size=6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "id": "914cd28d-6aa7-4614-ad3d-00ac271fdb3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_status_bar(iteration, total, loss, metrics=None, size=30):\n",
    "    metrics = \" - \".join([\"{}:, {:.4f}\".format(m.name, m.result())\n",
    "                          for m in [loss] + (metrics or [])])\n",
    "    end = \"\" if iteration < total else \"\\n\"\n",
    "    print(\"\\r{} - {}\".format(progress_bar(iteration, total), metrics), end=end)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "id": "30af90e3-77be-46c4-96c3-13bc48693836",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "50/50 [==============================] - loss:, 0.0900 - mean_square:, 858.5000\n"
     ]
    }
   ],
   "source": [
    "mean_loss = keras.metrics.Mean(name=\"loss\")\n",
    "mean_square = keras.metrics.Mean(name=\"mean_square\")\n",
    "for i in range(1, 50 + 1):\n",
    "    loss = 1 / i\n",
    "    mean_loss(loss)\n",
    "    mean_square(i**2)\n",
    "    print_status_bar(i, 50, mean_loss, [mean_square])\n",
    "    time.sleep(0.05)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "d1b8abd7-807f-4208-8ce4-c5f906b805db",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "900c8538-3b5e-4ca2-957d-d325354aa03d",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_epochs = 5\n",
    "batch_size = 32\n",
    "n_steps = len(X_train) // batch_size\n",
    "optimizer = keras.optimizers.Nadam(learning_rate=0.01)\n",
    "loss_fn = keras.losses.mean_squared_error\n",
    "mean_loss = keras.metrics.Mean()\n",
    "metrics = [keras.metrics.MeanAbsoluteError()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "id": "0e77ba99-d50f-4d72-a632-8ce886174f44",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "11610/11610 [==============================] - mean:, 1.3431 - mean_absolute_error:, 0.5895\n",
      "Epoch 2/5\n",
      "11610/11610 [==============================] - mean:, 0.7034 - mean_absolute_error:, 0.5519\n",
      "Epoch 3/5\n",
      "11610/11610 [==============================] - mean:, 0.6928 - mean_absolute_error:, 0.5532\n",
      "Epoch 4/5\n",
      "11610/11610 [==============================] - mean:, 0.6798 - mean_absolute_error:, 0.5459\n",
      "Epoch 5/5\n",
      "11610/11610 [==============================] - mean:, 0.6880 - mean_absolute_error:, 0.5510\n"
     ]
    }
   ],
   "source": [
    "for epoch in range(1, n_epochs + 1):\n",
    "    print(\"Epoch {}/{}\".format(epoch, n_epochs))\n",
    "    for step in range(1, n_steps + 1):\n",
    "        X_batch, y_batch = random_batch(X_train_scaled, y_train)\n",
    "        with tf.GradientTape() as tape:\n",
    "            y_pred = model(X_batch)\n",
    "            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))\n",
    "            loss = tf.add_n([main_loss] + model.losses)\n",
    "        gradients = tape.gradient(loss, model.trainable_variables)\n",
    "        optimizer.apply_gradients(zip(gradients, model.trainable_variables))\n",
    "        for variable in model.variables:\n",
    "            if variable.constraint is not None:\n",
    "                variable.assign(variable.constraint(variable))\n",
    "        mean_loss(loss)\n",
    "        for metric in metrics:\n",
    "            metric(y_batch, y_pred)\n",
    "        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)\n",
    "    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)\n",
    "    for metric in [mean_loss] + metrics:\n",
    "        metric.reset_states()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "id": "d82c6164-b444-4eaa-8e40-7931888afd80",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ceffaccdbb20484298d7a22566a13b29",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "All epochs:   0%|          | 0/5 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15b1a09683d14861b128277e3d004684",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Epoch 1/5:   0%|          | 0/362 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e61cd4c78e57432f9e5f297a8a5498c2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Epoch 2/5:   0%|          | 0/362 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0ae38ee903244f10a634d580ded97704",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Epoch 3/5:   0%|          | 0/362 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "542d1070bbea4ebca5a0bb6f2d2f6c70",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Epoch 4/5:   0%|          | 0/362 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "df12d08b70dc470eba0900e61550fbbd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Epoch 5/5:   0%|          | 0/362 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "try:\n",
    "    from tqdm.notebook import trange\n",
    "    from collections import OrderedDict\n",
    "    with trange(1, n_epochs + 1, desc=\"All epochs\") as epochs:\n",
    "        for epoch in epochs:\n",
    "            with trange(1, n_steps + 1, desc=\"Epoch {}/{}\".format(epoch, n_epochs)) as steps:\n",
    "                for step in steps:\n",
    "                    X_batch, y_batch = random_batch(X_train_scaled, y_train)\n",
    "                    with tf.GradientTape() as tape:\n",
    "                        y_pred = model(X_batch)\n",
    "                        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))\n",
    "                        loss = tf.add_n([main_loss] + model.losses)\n",
    "                    gradients = tape.gradient(loss, model.trainable_variables)\n",
    "                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))\n",
    "                    for variable in model.variables:\n",
    "                        if variable.constraint is not None:\n",
    "                            variable.assign(variable.constraint(variable))\n",
    "                    status = OrderedDict()\n",
    "                    mean_loss(loss)\n",
    "                    status[\"loss\"] = mean_loss.result().numpy()\n",
    "                    for metric in metrics:\n",
    "                        metric(y_batch, y_pred)\n",
    "                        status[metric.name] = metric.result().numpy()\n",
    "                    steps.set_postfix(status)\n",
    "                for metric in [mean_loss] + metrics:\n",
    "                    metric.reset_states()\n",
    "except ImportError as ex:\n",
    "    print(\"To run this cell, please install tqdm, ipywidgets and restart Jupyter\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc946b5c-552b-4cd2-9cdf-69c51e747e6d",
   "metadata": {},
   "source": [
    "## TensorFlow Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "id": "35a7bd79-eaaf-449c-8eea-acb8afb2b8a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def cube(X):\n",
    "    return X ** 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "id": "271c500b-3af4-4d47-a9d4-b11ad3d897fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 209,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cube(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "id": "f78b83ba-a66a-4426-8ba9-f16e8cfdbbd6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=8.0>"
      ]
     },
     "execution_count": 210,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cube(tf.constant(2.0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "id": "b1527f4c-8cd7-4927-b615-081ae8b99275",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.eager.def_function.Function at 0x7f87bc5eab90>"
      ]
     },
     "execution_count": 211,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf_cube = tf.function(cube)\n",
    "tf_cube"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "id": "b56787cf-0f3d-4733-896d-842e869ceff9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=8>"
      ]
     },
     "execution_count": 212,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf_cube(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "id": "99520b0d-169c-4002-839f-e1f71479a381",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=8.0>"
      ]
     },
     "execution_count": 213,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf_cube(tf.constant(2.0))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a48d729-35de-4552-8f07-08b4d903348b",
   "metadata": {},
   "source": [
    "### TF Functions and Concrete Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "id": "b1c191df-b4cb-47c9-985e-c71d0a615da8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.framework.func_graph.FuncGraph at 0x7f87b6e0d570>"
      ]
     },
     "execution_count": 214,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function = tf_cube.get_concrete_function(tf.constant(2.0))\n",
    "concrete_function.graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "id": "8c3d4596-d003-4403-9123-e1df77c92cd9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=8.0>"
      ]
     },
     "execution_count": 215,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function(tf.constant(2.0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "id": "a059b0eb-c0f8-4fee-90e8-f22e808b5918",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 216,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function is tf_cube.get_concrete_function(tf.constant(2.0))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4d9859c-e6f2-459e-99f9-3cc3046329f1",
   "metadata": {},
   "source": [
    "### Exploring Function Definitions and Graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "id": "4cc7abda-1def-4cb7-b5bb-5084f82367a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.framework.func_graph.FuncGraph at 0x7f87b6e0d570>"
      ]
     },
     "execution_count": 217,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function.graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "id": "7bc574ec-84bb-48e8-875d-d3a7dcdd9250",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Operation 'X' type=Placeholder>,\n",
       " <tf.Operation 'pow/y' type=Const>,\n",
       " <tf.Operation 'pow' type=Pow>,\n",
       " <tf.Operation 'Identity' type=Identity>]"
      ]
     },
     "execution_count": 218,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ops = concrete_function.graph.get_operations()\n",
    "ops"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "id": "27ef12fe-6f23-4e81-b4ad-8c90170b04b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor 'X:0' shape=() dtype=float32>,\n",
       " <tf.Tensor 'pow/y:0' shape=() dtype=float32>]"
      ]
     },
     "execution_count": 219,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pow_op = ops[2]\n",
    "list(pow_op.inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "id": "dff7a871-3c44-4e6d-b451-6b0b15a539ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Tensor 'pow:0' shape=() dtype=float32>]"
      ]
     },
     "execution_count": 220,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pow_op.outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "id": "90c152c1-1d7a-4fd4-b006-eda5ecf5aaef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Operation 'X' type=Placeholder>"
      ]
     },
     "execution_count": 221,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function.graph.get_operation_by_name('X')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "id": "badc12c7-2784-48d8-9cef-eb05c1904309",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor 'Identity:0' shape=() dtype=float32>"
      ]
     },
     "execution_count": 222,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function.graph.get_tensor_by_name('Identity:0')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "id": "e9409167-e461-4ffe-b73b-a432ae302f88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "name: \"__inference_cube_1083107\"\n",
       "input_arg {\n",
       "  name: \"x\"\n",
       "  type: DT_FLOAT\n",
       "}\n",
       "output_arg {\n",
       "  name: \"identity\"\n",
       "  type: DT_FLOAT\n",
       "}"
      ]
     },
     "execution_count": 223,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concrete_function.function_def.signature"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "39ba0aed-f60c-4864-8fc3-a4dfc24dedaa",
   "metadata": {},
   "source": [
    "### How TF Functions Trace Python Functions to Extract Their Computation Graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "id": "29ee7283-f021-41ca-a1f5-22fe55970ee2",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function\n",
    "def tf_cube(x):\n",
    "    print(\"print:\", x)\n",
    "    return x ** 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "id": "4d4e12ec-1d17-469f-a780-aa0d2cbca1cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "print: Tensor(\"x:0\", shape=(), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "result = tf_cube(tf.constant(2.0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "id": "5f6b9451-2867-4c91-8586-0dfd8cfa21e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=float32, numpy=8.0>"
      ]
     },
     "execution_count": 226,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "id": "08ab2fc1-1f43-48df-9c83-f0d21a992570",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "print: 2\n",
      "print: 3\n",
      "print: Tensor(\"x:0\", shape=(1, 2), dtype=float32)\n",
      "print: Tensor(\"x:0\", shape=(2, 2), dtype=float32)\n",
      "WARNING:tensorflow:5 out of the last 5 calls to <function tf_cube at 0x7f87b6e29bd0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:5 out of the last 5 calls to <function tf_cube at 0x7f87b6e29bd0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "print: Tensor(\"x:0\", shape=(3, 2), dtype=float32)\n",
      "WARNING:tensorflow:6 out of the last 6 calls to <function tf_cube at 0x7f87b6e29bd0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:6 out of the last 6 calls to <function tf_cube at 0x7f87b6e29bd0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n"
     ]
    }
   ],
   "source": [
    "result = tf_cube(2)\n",
    "result = tf_cube(3)\n",
    "result = tf_cube(tf.constant([[1., 2.]]))\n",
    "result = tf_cube(tf.constant([[3., 4.], [5., 6.]]))\n",
    "result = tf_cube(tf.constant([[7., 8.], [9., 10.], [11., 12.]]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a3a835a-ed32-419e-9d85-dc495f65a002",
   "metadata": {},
   "source": [
    "It is also possible to specify a particular input signature:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 228,
   "id": "0b5546af-79cb-4897-b3d3-3701f8051a97",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function(input_signature=[tf.TensorSpec([None, 28, 28], tf.float32)])\n",
    "def shrink(images):\n",
    "    print(\"Tracing\", images)\n",
    "    return images[:, ::2, ::2] # drop half the rows and columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 229,
   "id": "060f159c-6f5a-45f8-a357-ebff4e354f8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "id": "33aefa33-4a4c-49cf-9ce6-08885675074f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tracing Tensor(\"images:0\", shape=(None, 28, 28), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "img_batch_1 = tf.random.uniform(shape=[100, 28, 28])\n",
    "img_batch_2 = tf.random.uniform(shape=[50, 28, 28])\n",
    "preprocessed_images = shrink(img_batch_1) # Traces the function.\n",
    "preprocessed_images = shrink(img_batch_2) # Reuses the same concrete function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 231,
   "id": "2090a077-0c39-417f-b169-c17a23fa605f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Python inputs incompatible with input_signature:\n",
      "  inputs: (\n",
      "    tf.Tensor(\n",
      "[[[0.11529493 0.1944716 ]\n",
      "  [0.01845205 0.5866661 ]]\n",
      "\n",
      " [[0.12523496 0.42132652]\n",
      "  [0.21750104 0.6749774 ]]], shape=(2, 2, 2), dtype=float32))\n",
      "  input_signature: (\n",
      "    TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name=None)).\n"
     ]
    }
   ],
   "source": [
    "img_batch_3 = tf.random.uniform(shape=[2, 2, 2])\n",
    "try:\n",
    "    preprocessed_images = shrink(img_batch_3) # rejects unexpected types or shapes\n",
    "except ValueError as ex:\n",
    "    print(ex)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b1025a7-134e-4d61-8c01-046372c2c24a",
   "metadata": {},
   "source": [
    "### Using Autograph To Capture Control Flow"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b03764b3-d2b0-45ac-964b-a3757f4e9cd9",
   "metadata": {},
   "source": [
    "A \"static\" `for` loop using `range()`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 232,
   "id": "f73afdba-435c-4c56-821c-f54fb80f40d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function\n",
    "def add_10(x):\n",
    "    for i in range(10):\n",
    "        x +=1\n",
    "    return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "id": "4dbc6c48-8f6f-4b84-ab90-138cd710a78d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=15>"
      ]
     },
     "execution_count": 233,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "add_10(tf.constant(5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "id": "c5f8b3ed-c10a-4a1e-a10a-1826d3f3c013",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Operation 'x' type=Placeholder>,\n",
       " <tf.Operation 'add/y' type=Const>,\n",
       " <tf.Operation 'add' type=AddV2>,\n",
       " <tf.Operation 'add_1/y' type=Const>,\n",
       " <tf.Operation 'add_1' type=AddV2>,\n",
       " <tf.Operation 'add_2/y' type=Const>,\n",
       " <tf.Operation 'add_2' type=AddV2>,\n",
       " <tf.Operation 'add_3/y' type=Const>,\n",
       " <tf.Operation 'add_3' type=AddV2>,\n",
       " <tf.Operation 'add_4/y' type=Const>,\n",
       " <tf.Operation 'add_4' type=AddV2>,\n",
       " <tf.Operation 'add_5/y' type=Const>,\n",
       " <tf.Operation 'add_5' type=AddV2>,\n",
       " <tf.Operation 'add_6/y' type=Const>,\n",
       " <tf.Operation 'add_6' type=AddV2>,\n",
       " <tf.Operation 'add_7/y' type=Const>,\n",
       " <tf.Operation 'add_7' type=AddV2>,\n",
       " <tf.Operation 'add_8/y' type=Const>,\n",
       " <tf.Operation 'add_8' type=AddV2>,\n",
       " <tf.Operation 'add_9/y' type=Const>,\n",
       " <tf.Operation 'add_9' type=AddV2>,\n",
       " <tf.Operation 'Identity' type=Identity>]"
      ]
     },
     "execution_count": 234,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "add_10.get_concrete_function(tf.constant(5)).graph.get_operations()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f25764db-e202-4bdd-b09d-9a2f9839121e",
   "metadata": {},
   "source": [
    "A \"dynamic\" loop using `tf.while_loop()`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "id": "137ef459-0d64-43b7-a459-00238c82e418",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function\n",
    "def add_10(x):\n",
    "    condition = lambda i, x: tf.less(i, 10)\n",
    "    body = lambda i, x: (tf.add(i, 1), tf.add(x, 1))\n",
    "    final_i, final_x = tf.while_loop(condition, body, [tf.constant(0), x])\n",
    "    return final_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 236,
   "id": "528285bb-f3ba-445b-8057-518304f77bcb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=15>"
      ]
     },
     "execution_count": 236,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "add_10(tf.constant(5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "id": "08f754a8-9d41-4327-b207-0fce13090000",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Operation 'x' type=Placeholder>,\n",
       " <tf.Operation 'Const' type=Const>,\n",
       " <tf.Operation 'while/maximum_iterations' type=Const>,\n",
       " <tf.Operation 'while/loop_counter' type=Const>,\n",
       " <tf.Operation 'while' type=StatelessWhile>,\n",
       " <tf.Operation 'Identity' type=Identity>]"
      ]
     },
     "execution_count": 237,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "add_10.get_concrete_function(tf.constant(5)).graph.get_operations()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d416637-2b33-4ea1-a0e3-f961710942a8",
   "metadata": {},
   "source": [
    "A \"dynamic\" `for` loop using `tf.range()` (captured by autograph):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "id": "22427199-d36b-47ba-b0ad-8ef66fffaf84",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tf.function\n",
    "def add_10(x):\n",
    "    for i in tf.range(10):\n",
    "        x = x + 1\n",
    "    return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "id": "6cc15701-58a9-4624-b999-e0f8c283cb70",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<tf.Operation 'x' type=Placeholder>,\n",
       " <tf.Operation 'range/start' type=Const>,\n",
       " <tf.Operation 'range/limit' type=Const>,\n",
       " <tf.Operation 'range/delta' type=Const>,\n",
       " <tf.Operation 'range' type=Range>,\n",
       " <tf.Operation 'sub' type=Sub>,\n",
       " <tf.Operation 'floordiv' type=FloorDiv>,\n",
       " <tf.Operation 'mod' type=FloorMod>,\n",
       " <tf.Operation 'zeros_like' type=Const>,\n",
       " <tf.Operation 'NotEqual' type=NotEqual>,\n",
       " <tf.Operation 'Cast' type=Cast>,\n",
       " <tf.Operation 'add' type=AddV2>,\n",
       " <tf.Operation 'zeros_like_1' type=Const>,\n",
       " <tf.Operation 'Maximum' type=Maximum>,\n",
       " <tf.Operation 'while/maximum_iterations' type=Const>,\n",
       " <tf.Operation 'while/loop_counter' type=Const>,\n",
       " <tf.Operation 'while' type=StatelessWhile>,\n",
       " <tf.Operation 'Identity' type=Identity>]"
      ]
     },
     "execution_count": 239,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "add_10.get_concrete_function(tf.constant(0)).graph.get_operations()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eaa1b4c9-e534-47ed-8f00-f97adde05a2a",
   "metadata": {},
   "source": [
    "### Handling Variables and Other Resources in TF Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 240,
   "id": "8834c560-16f3-43ef-9570-b181f1eea0ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "counter = tf.Variable(0)\n",
    "\n",
    "@tf.function\n",
    "def increment(counter, c=1):\n",
    "    return counter.assign_add(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 241,
   "id": "9e0743c4-67d3-4daf-a9c0-a704a7244b73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=2>"
      ]
     },
     "execution_count": 241,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "increment(counter)\n",
    "increment(counter)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "id": "ac05d341-2466-4d00-8f93-2899fb48dfbb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "name: \"counter\"\n",
       "type: DT_RESOURCE"
      ]
     },
     "execution_count": 242,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function_def = increment.get_concrete_function(counter).function_def\n",
    "function_def.signature.input_arg[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "id": "5ebce34e-9b7a-45af-838e-917c130a94f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "counter = tf.Variable(0)\n",
    "\n",
    "@tf.function\n",
    "def increment(c=1):\n",
    "    return counter.assign_add(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 244,
   "id": "9a0831f2-7097-4d58-bfbe-bd24e5991da9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=2>"
      ]
     },
     "execution_count": 244,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "increment()\n",
    "increment()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 245,
   "id": "cc024662-f966-4ae3-8334-9f0466383e09",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "name: \"assignaddvariableop_resource\"\n",
       "type: DT_RESOURCE"
      ]
     },
     "execution_count": 245,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function_def = increment.get_concrete_function().function_def\n",
    "function_def.signature.input_arg[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 246,
   "id": "45c8246c-3c05-40eb-8b2d-cfc95be2d5a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Counter:\n",
    "    def __init__(self):\n",
    "        self.counter = tf.Variable(0)\n",
    "        \n",
    "    @tf.function\n",
    "    def increment(self, c=1):\n",
    "        return self.counter.assign_add(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "id": "41099347-85e0-4704-9aa8-5916518cff88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(), dtype=int32, numpy=2>"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c = Counter()\n",
    "c.increment()\n",
    "c.increment()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "id": "1a44ed96-cce0-4242-b084-ad50b5efe718",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "def tf__add(x):\n",
      "    with ag__.FunctionScope('add_10', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:\n",
      "        do_return = False\n",
      "        retval_ = ag__.UndefinedReturnValue()\n",
      "\n",
      "        def get_state():\n",
      "            return (x,)\n",
      "\n",
      "        def set_state(vars_):\n",
      "            nonlocal x\n",
      "            (x,) = vars_\n",
      "\n",
      "        def loop_body(itr):\n",
      "            nonlocal x\n",
      "            i = itr\n",
      "            x = ag__.ld(x)\n",
      "            x += 1\n",
      "        i = ag__.Undefined('i')\n",
      "        ag__.for_stmt(ag__.converted_call(ag__.ld(tf).range, (10,), None, fscope), None, loop_body, get_state, set_state, ('x',), {'iterate_names': 'i'})\n",
      "        try:\n",
      "            do_return = True\n",
      "            retval_ = ag__.ld(x)\n",
      "        except:\n",
      "            do_return = False\n",
      "            raise\n",
      "        return fscope.ret(retval_, do_return)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "@tf.function\n",
    "def add_10(x):\n",
    "    for i in tf.range(10):\n",
    "        x +=1\n",
    "    return x\n",
    "\n",
    "print(tf.autograph.to_code(add_10.python_function))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 249,
   "id": "b045b47f-9b39-4602-a96e-a2f172372cb0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_tf_code(func):\n",
    "    from IPython.display import display, Markdown\n",
    "    if hasattr(func, \"python_function\"):\n",
    "        func = func.python_function\n",
    "    code = tf.autograph.to_code(func)\n",
    "    display(Markdown('```python\\n{}\\n```'.format(code)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "id": "a9a0d1ea-0f6f-48ab-ba27-ea3d6a7da3b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "```python\n",
       "def tf__add(x):\n",
       "    with ag__.FunctionScope('add_10', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:\n",
       "        do_return = False\n",
       "        retval_ = ag__.UndefinedReturnValue()\n",
       "\n",
       "        def get_state():\n",
       "            return (x,)\n",
       "\n",
       "        def set_state(vars_):\n",
       "            nonlocal x\n",
       "            (x,) = vars_\n",
       "\n",
       "        def loop_body(itr):\n",
       "            nonlocal x\n",
       "            i = itr\n",
       "            x = ag__.ld(x)\n",
       "            x += 1\n",
       "        i = ag__.Undefined('i')\n",
       "        ag__.for_stmt(ag__.converted_call(ag__.ld(tf).range, (10,), None, fscope), None, loop_body, get_state, set_state, ('x',), {'iterate_names': 'i'})\n",
       "        try:\n",
       "            do_return = True\n",
       "            retval_ = ag__.ld(x)\n",
       "        except:\n",
       "            do_return = False\n",
       "            raise\n",
       "        return fscope.ret(retval_, do_return)\n",
       "\n",
       "```"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display_tf_code(add_10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "843c6fe1-4a08-4ede-8428-dfd88d3fa872",
   "metadata": {},
   "source": [
    "## Using TF Functions with tf.keras (or Not)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b395f84-b7e9-4cc2-8570-e9de6ffc550e",
   "metadata": {},
   "source": [
    "By default, tf.keras will automatically convert your custom code into TF Functions, no need to use\n",
    "`tf.function()`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 251,
   "id": "f8c998cd-5101-47ac-9b60-7f2a88fcca25",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom loss function\n",
    "def my_mse(y_true, y_pred):\n",
    "    print(\"Tracing loss my_mse()\")\n",
    "    return tf.reduce_mean(tf.square(y_pred - y_true))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "id": "5eec2093-2a30-4d5d-8b1a-d70e8d7f4f21",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom metric function\n",
    "def my_mae(y_true, y_pred):\n",
    "    print(\"Tracing metric my_mae()\")\n",
    "    return tf.reduce_mean(tf.abs(y_pred - y_true))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "id": "cd862a3a-f0d0-4ca9-bb7d-29ed195389ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom layer\n",
    "class MyDense(keras.layers.Layer):\n",
    "    def __init__(self, units, activation=None, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units = units\n",
    "        self.activation = keras.activations.get(activation)\n",
    "        \n",
    "    def build(self, input_shape):\n",
    "        self.kernel = self.add_weight(name='kernel',\n",
    "                                      shape=(input_shape[1], self.units),\n",
    "                                      initializer='uniform',\n",
    "                                      trainable=True)\n",
    "        self.biases = self.add_weight(name='bias',\n",
    "                                      shape=(self.units),\n",
    "                                      initializer='zeros',\n",
    "                                      trainable=True)\n",
    "        super().build(input_shape)\n",
    "        \n",
    "    def call(self, X):\n",
    "        print(\"Tracing MyDense.call()\")\n",
    "        return self.activation(X @ self.kernel + self.biases)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "id": "79cfc95c-5671-4cd8-8f50-a0de3dd119f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "id": "b15582c7-468b-48e0-ba89-1b79a08eb78a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom model\n",
    "class MyModel(keras.models.Model):\n",
    "    def __init__(self, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.hidden1 = MyDense(30, activation=\"relu\")\n",
    "        self.hidden2 = MyDense(30, activation=\"relu\")\n",
    "        self.output_ = MyDense(1)\n",
    "        \n",
    "    def call(self, input):\n",
    "        print(\"Tracing MyModel.call()\")\n",
    "        hidden1 = self.hidden1(input)\n",
    "        hidden2 = self.hidden2(hidden1)\n",
    "        concat = keras.layers.concatenate([input, hidden2])\n",
    "        output = self.output_(concat)\n",
    "        return output\n",
    "    \n",
    "model = MyModel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "id": "0e726b79-ec10-449c-b0ed-ede7dd256e6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=my_mse, optimizer=\"nadam\", metrics=[my_mae])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "id": "7eeefb8f-c413-4c85-b6be-e8b5347f67dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "346/363 [===========================>..] - ETA: 0s - loss: 1.4020 - my_mae: 0.8311Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "363/363 [==============================] - 4s 7ms/step - loss: 1.3652 - my_mae: 0.8171 - val_loss: 0.4983 - val_my_mae: 0.5173\n",
      "Epoch 2/2\n",
      "363/363 [==============================] - 2s 4ms/step - loss: 0.4722 - my_mae: 0.4887 - val_loss: 0.4161 - val_my_mae: 0.4670\n",
      "162/162 [==============================] - 0s 2ms/step - loss: 0.4083 - my_mae: 0.4628\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.4083360433578491, 0.46279317140579224]"
      ]
     },
     "execution_count": 257,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled, y_train, epochs=2,\n",
    "          validation_data=(X_valid_scaled, y_valid))\n",
    "model.evaluate(X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e45512aa-0993-47b5-b8b0-6ad4ce4c225b",
   "metadata": {},
   "source": [
    "You can turn this off by creating the model with `dynamic=True` (or calling `super().__init__(dynamic=True, **kwargs)` in the model's constructor):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 258,
   "id": "19581e3f-957e-4fb6-b625-c42147ae6730",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "id": "0acdc360-3e78-4619-8ccc-fd4a3a03313a",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = MyModel(dynamic=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 260,
   "id": "8583267e-e5f6-4324-b777-7232e44796a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=my_mse, optimizer=\"nadam\", metrics=[my_mae])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18d09c58-455d-4844-b8ee-6f95f5118001",
   "metadata": {},
   "source": [
    "Note the custom code will be called at each iteration. Let's fit, validate and evaluate with tiny datasets to avoid getting too much output:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 261,
   "id": "3db23ea3-4b8d-4402-880b-3f445b92f770",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[6.788027286529541, 2.277451753616333]"
      ]
     },
     "execution_count": 261,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled[:64], y_train[:64], epochs=1,\n",
    "          validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)\n",
    "model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "119584d2-f535-4ef0-abd3-78a2064f69b3",
   "metadata": {},
   "source": [
    "Alternatively, you can compile a model with `run_eagerly=True`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "id": "011e0dfc-264c-4bc9-a125-d77ca7eec184",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 263,
   "id": "0288e0b4-bbb5-45f5-9649-5332b2823420",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = MyModel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 264,
   "id": "b7f2f798-b48f-4ec9-95d8-a1b58555f4af",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss=my_mse, optimizer=\"nadam\", metrics=[my_mae], run_eagerly=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 265,
   "id": "d9d00365-f119-41b8-9e98-05f77e20932c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n",
      "Tracing MyModel.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing MyDense.call()\n",
      "Tracing loss my_mse()\n",
      "Tracing metric my_mae()\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[6.438323974609375, 2.2199299335479736]"
      ]
     },
     "execution_count": 265,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train_scaled[:64], y_train[:64], epochs=1,\n",
    "          validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)\n",
    "model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "39b5722e-f91f-4f62-9571-02ed4a5f2d32",
   "metadata": {},
   "source": [
    "## Custom Optimizers"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d7392b6-b0ee-4665-8517-506d9d12a22e",
   "metadata": {},
   "source": [
    "Defining custom optimizers is not very common, but in case you are one of the happy few who gets to write one, here is an example:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "id": "4d83df6f-b81c-492a-ad69-67755c06d4c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "class MyMomentumOptimizer(keras.optimizers.Optimizer):\n",
    "    def __init__(self, learning_rate=0.001, momentum=0.9, name=\"MyMomentumOptimizer\", **kwargs):\n",
    "        \"\"\"Call super().__init__() and use _set_hyper() to store hyperparameters\"\"\"\n",
    "        super().__init__(name, **kwargs)\n",
    "        self._set_hyper(\"learning_rate\", kwargs.get(\"lr\", learning_rate)) # handle lr=learning_rate\n",
    "        self._set_hyper(\"decay\", self._initial_decay) # \n",
    "        self._set_hyper(\"momentum\", momentum)\n",
    "    \n",
    "    def _create_slots(self, var_list):\n",
    "        \"\"\"For each model variable, create the optimizer variable associated with it.\n",
    "        TensorFlow calls these optimizer variables \"slots\".\n",
    "        For momentum optimization, we need one momentum slot per model variable.\n",
    "        \"\"\"\n",
    "        for var in var_list:\n",
    "            self.add_slot(var, \"momentum\")\n",
    "\n",
    "    @tf.function\n",
    "    def _resource_apply_dense(self, grad, var):\n",
    "        \"\"\"Update the slots and perform one optimization step for one model variable\n",
    "        \"\"\"\n",
    "        var_dtype = var.dtype.base_dtype\n",
    "        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay\n",
    "        momentum_var = self.get_slot(var, \"momentum\")\n",
    "        momentum_hyper = self._get_hyper(\"momentum\", var_dtype)\n",
    "        momentum_var.assign(momentum_var * momentum_hyper - (1. - momentum_hyper)* grad)\n",
    "        var.assign_add(momentum_var * lr_t)\n",
    "\n",
    "    def _resource_apply_sparse(self, grad, var):\n",
    "        raise NotImplementedError\n",
    "\n",
    "    def get_config(self):\n",
    "        base_config = super().get_config()\n",
    "        return {\n",
    "            **base_config,\n",
    "            \"learning_rate\": self._serialize_hyperparameter(\"learning_rate\"),\n",
    "            \"decay\": self._serialize_hyperparameter(\"decay\"),\n",
    "            \"momentum\": self._serialize_hyperparameter(\"momentum\"),\n",
    "        }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "id": "51ec8dc4-b09a-4d1d-8a6c-f46a99c3073c",
   "metadata": {},
   "outputs": [],
   "source": [
    "keras.backend.clear_session()\n",
    "np.random.seed(100)\n",
    "tf.random.set_seed(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 268,
   "id": "9a2defec-add6-428d-a446-c3af6928a97b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "363/363 [==============================] - 1s 3ms/step - loss: 4.4177\n",
      "Epoch 2/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 1.4329\n",
      "Epoch 3/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.8462\n",
      "Epoch 4/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.7021\n",
      "Epoch 5/5\n",
      "363/363 [==============================] - 1s 2ms/step - loss: 0.6577\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f87bc8a1cf0>"
      ]
     },
     "execution_count": 268,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[8])])\n",
    "model.compile(loss=\"mse\", optimizer=MyMomentumOptimizer())\n",
    "model.fit(X_train_scaled, y_train, epochs=5)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
