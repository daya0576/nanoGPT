{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "977bc155",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.doubanio.com/simple\n",
      "Requirement already satisfied: torch in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (2.0.1)\n",
      "Requirement already satisfied: torchvision in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (0.15.2)\n",
      "Requirement already satisfied: torchaudio in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (2.0.2)\n",
      "Requirement already satisfied: jinja2 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torch) (3.1.2)\n",
      "Requirement already satisfied: typing-extensions in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torch) (4.7.0)\n",
      "Requirement already satisfied: networkx in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torch) (3.1)\n",
      "Requirement already satisfied: filelock in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torch) (3.12.0)\n",
      "Requirement already satisfied: sympy in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torch) (1.12)\n",
      "Requirement already satisfied: requests in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torchvision) (2.31.0)\n",
      "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torchvision) (9.5.0)\n",
      "Requirement already satisfied: numpy in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from torchvision) (1.25.2)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from jinja2->torch) (2.1.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from requests->torchvision) (2023.5.7)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from requests->torchvision) (3.1.0)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from requests->torchvision) (2.0.3)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from requests->torchvision) (3.4)\n",
      "Requirement already satisfied: mpmath>=0.19 in /Users/henry/.pyenv/versions/3.9.16/lib/python3.9/site-packages (from sympy->torch) (1.3.0)\n",
      "\u001b[33mWARNING: You are using pip version 22.0.4; however, version 23.1.2 is available.\n",
      "You should consider upgrading via the '/Users/henry/.pyenv/versions/3.9.16/bin/python3.9 -m pip install --upgrade pip' command.\u001b[0m\u001b[33m\n",
      "\u001b[0m"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'torch'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[2], line 4\u001b[0m\n\u001b[1;32m      1\u001b[0m get_ipython()\u001b[38;5;241m.\u001b[39msystem(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpyenv local 3.9\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m      2\u001b[0m get_ipython()\u001b[38;5;241m.\u001b[39msystem(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpip3 install torch torchvision torchaudio\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m----> 4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'torch'"
     ]
    }
   ],
   "source": [
    "!pip3 install torch torchvision torchaudio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2fd7454f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "start downloading...\n",
      "--2023-08-20 15:08:48--  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\n",
      "wget: /Users/henry/.netrc:1: unknown token \"undefined\"\n",
      "Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 198.18.1.89\n",
      "Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|198.18.1.89|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 1115394 (1.1M) [text/plain]\n",
      "Saving to: ‘input.txt.7’\n",
      "\n",
      "input.txt.7         100%[===================>]   1.06M  1.99MB/s    in 0.5s    \n",
      "\n",
      "2023-08-20 15:08:48 (1.99 MB/s) - ‘input.txt.7’ saved [1115394/1115394]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 数据集下载\n",
    "!echo \"start downloading...\"\n",
    "!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "66f11fca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "length of dataset in characters:  1115394\n",
      "First Citizen:\n",
      "Before we proceed any further, hear me speak.\n",
      "\n",
      "All:\n",
      "Speak, speak.\n",
      "\n",
      "First Citizen:\n",
      "You\n"
     ]
    }
   ],
   "source": [
    "# read it in to inspect it\n",
    "with open('input.txt', 'r', encoding='utf-8') as f:\n",
    "    text = f.read()\n",
    "print(\"length of dataset in characters: \", len(text))\n",
    "# let's look at the first 1000 characters\n",
    "print(text[:100])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0a5bcfcf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n",
      "65\n"
     ]
    }
   ],
   "source": [
    "# here are all the unique characters that occur in this text\n",
    "chars = sorted(list(set(text)))\n",
    "vocab_size = len(chars)\n",
    "print(''.join(chars))\n",
    "print(vocab_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6d704e68",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[46, 47, 47, 1, 58, 46, 43, 56, 43]\n",
      "hii there\n"
     ]
    }
   ],
   "source": [
    "# create a mapping from characters to integers\n",
    "stoi = { ch:i for i,ch in enumerate(chars) }\n",
    "itos = { i:ch for i,ch in enumerate(chars) }\n",
    "encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers\n",
    "decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string\n",
    "\n",
    "print(encode(\"hii there\"))\n",
    "print(decode(encode(\"hii there\")))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "24fd11cb",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'torch'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[5], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# let's now encode the entire text dataset and store it into a torch.Tensor\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mtorch\u001b[39;00m \u001b[38;5;66;03m# we use PyTorch: https://pytorch.org\u001b[39;00m\n\u001b[1;32m      3\u001b[0m data \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mtensor(encode(text), dtype\u001b[38;5;241m=\u001b[39mtorch\u001b[38;5;241m.\u001b[39mlong)\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28mprint\u001b[39m(data\u001b[38;5;241m.\u001b[39mshape, data\u001b[38;5;241m.\u001b[39mdtype)\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'torch'"
     ]
    }
   ],
   "source": [
    "# let's now encode the entire text dataset and store it into a torch.Tensor\n",
    "import torch # we use PyTorch: https://pytorch.org\n",
    "data = torch.tensor(encode(text), dtype=torch.long)\n",
    "print(data.shape, data.dtype)\n",
    "print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this\n",
    "\n",
    "# 转化为多维矩阵\n",
    "# 0: new line\n",
    "# 1: space \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "44b64520",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'data' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[4], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# train and validation sets -> 用于检测是否 overfitting\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m n \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mint\u001b[39m(\u001b[38;5;241m0.9\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;28mlen\u001b[39m(\u001b[43mdata\u001b[49m)) \u001b[38;5;66;03m# first 90% will be train, rest val\u001b[39;00m\n\u001b[1;32m      3\u001b[0m train_data \u001b[38;5;241m=\u001b[39m data[:n]\n\u001b[1;32m      4\u001b[0m val_data \u001b[38;5;241m=\u001b[39m data[n:] \u001b[38;5;66;03m# hide\u001b[39;00m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'data' is not defined"
     ]
    }
   ],
   "source": [
    "# train and validation sets -> 用于检测是否 overfitting\n",
    "n = int(0.9*len(data)) # first 90% will be train, rest val\n",
    "train_data = data[:n]\n",
    "val_data = data[n:] # hide\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09a81784",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "block_size = 8\n",
    "train_data[:block_size+1]\n",
    "\n",
    "x = train_data[:block_size]\n",
    "y = train_data[1:block_size+1]\n",
    "for t in range(block_size):\n",
    "    context = x[:t+1]\n",
    "    target = y[t]\n",
    "    print(f\"when input is {context} the target: {target}\")\n",
    "    \n",
    "# make Transformer Network be used to seeing contexts\n",
    "# as little as one all the way to block size\n",
    "# 用于在后续的推理，不论是输入一个还是多个 token（如果超过则需要 truncate）\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "208058bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# cost function \n",
    "\n",
    "torch.manual_seed(1337)\n",
    "batch_size = 4 # how many independent sequences will we process in parallel?\n",
    "block_size = 8 # what is the maximum context length for predictions?\n",
    "\n",
    "def get_batch(split):\n",
    "    # generate a small batch of data of inputs x and targets y\n",
    "    data = train_data if split == 'train' else val_data\n",
    "    ix = torch.randint(len(data) - block_size, (batch_size,))\n",
    "    x = torch.stack([data[i:i+block_size] for i in ix])\n",
    "    y = torch.stack([data[i+1:i+block_size+1] for i in ix])\n",
    "    return x, y\n",
    "\n",
    "xb, yb = get_batch('train')\n",
    "print('inputs:')\n",
    "print(xb.shape)\n",
    "print(xb)\n",
    "print('targets:')\n",
    "print(yb.shape)\n",
    "print(yb)\n",
    "\n",
    "print('----')\n",
    "\n",
    "for b in range(batch_size): # batch dimension\n",
    "    for t in range(block_size): # time dimension\n",
    "        context = xb[b, :t+1]\n",
    "        target = yb[b,t]\n",
    "        print(f\"when input is {context.tolist()} the target: {target}\")"
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
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
