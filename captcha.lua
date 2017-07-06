local dir = 'dataset/'
local validationSize = 1000
local iterations = 30
local batchSize = 128
local sgd_config = {
      learningRate = 0.1,
      learningRateDecay = 5.0e-6,
      momentum = 0.9
   }

local data = require 'data';
data.storeXY(dir)
local X,Y = data.loadXY(dir)
local Xt,Yt,Xv,Yv = data.split(X,Y,validationSize)

local models = require 'models';
local net,ct = models.cnnModel()

local net = net:float()
local ct = ct:float()
local Xv = Xv:float()
local Yv = Yv:float()
local Xt = Xt:float()
local Yt = Yt:float()

local train = require 'train';

train.sgd(net,ct,Xt,Yt,Xv,Yv,iterations,sgd_config,batchSize)

torch.save(dir .. 'net.t7',net)
