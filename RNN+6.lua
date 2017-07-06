
-- coding: utf-8

-- In[3]:

data = require 'data'


-- In[2]:

dir = 'simple/'


-- In[4]:

X,Y = data.loadXY(dir)
data.convert(Y)


-- In[1]:

function repl(N)
    local net = nn.ConcatTable()
    for i=1,N do 
        net:add(nn.Identity())
    end
    return net
end


-- In[2]:

require 'nn'
require 'rnn'
net = nn.Sequential()
net:add(nn.Reshape(1,50,200))
net:add(nn.SpatialConvolution(1,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.View(64*1*6))
net:add(nn.Reshape(8,48))
net:add(nn.SplitTable(2,3))
enc = nn.Sequential()
enc:add(net)
enc:add(nn.Sequencer(nn.LSTM(48, 48)))
enc:add(nn.SelectTable(-1))
dec = nn.Sequential()
dec:add(enc)
dec:add(repl(7))

mlp = nn.Sequential()
       :add(nn.LSTM(48, 48))
       :add(nn.Linear(48, 36))
       :add(nn.LogSoftMax())

dec:add(nn.Sequencer(mlp))
--net:add(repl(7))


-- In[7]:

require 'cunn';
require 'nn';
require 'rnn';
dec = torch.load('seq88.t7')
dec = dec:cuda()


-- In[8]:

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion = criterion:cuda()


-- In[9]:

rutil = require 'rutil'


-- In[10]:

Xt,Yt,Xv,Yv = data.split(X,Y,1000)


-- In[11]:

tnet = nn.SplitTable(2,2):cuda()


-- In[12]:

rutil.valid(dec,Xv,Yv,512,tnet)


-- In[33]:


function rutil.kfacc(outputs,targets)
    local Y,y = nil,nil;
    local N = outputs[1]:size(1)
    local C = outputs[1]:size(2)
    for k=1,--outputs do 
        Y = Y and torch.cat(Y,outputs[k]:reshape(N,1,C),2) or outputs[k]:reshape(N,1,C)
        y = y and torch.cat(y,targets[k]:reshape(N,1),2) or targets[k]:reshape(N,1)
    end
    local t,idx = Y:max(3)
    return idx:squeeze():eq(y):sum(2):eq(--outputs):sum()
end

function rutil.kvalid(rnn,Xv,Yv,batchSize,tnet)
    local batchSize = batchSize or 16
    local acc = 0
    local acci = {}
    local Nv = Xv:size(1)
    rnn:evaluate()
    for i=1,Nv,batchSize do 
        xlua.progress(i/batchSize, Nv/batchSize)
        local j = math.min(Nv,i+batchSize-1)
        local Xb = Xv[{{i,j}}]:cuda()            
        local Yb = Yv[{{i,j}}]:cuda()
        local inputs = Xb
        local targets = tnet:forward(Yb)
        local outputs = rnn:forward(inputs)
        local aa,ai = rutil.kfacc2(outputs,targets)
        acc = acc + aa
        rnn:forget()
    end
    return (acc*100)/Nv,acci
end


-- In[16]:

rutil.train(dec,criterion,Xt,Yt,Xv,Yv,8,64,tnet,0.01)


-- In[17]:

rutil.train(dec,criterion,Xt,Yt,Xv,Yv,8,64,tnet,0.01)


-- In[ ]:

torch.save('seq88.t7',dec)


-- In[30]:

rutil.kvalid(dec,Xv,Yv,64,tnet)


-- In[18]:

rutil.train(dec,criterion,Xt,Yt,Xv,Yv,8,64,tnet,0.1)


-- In[22]:

rutil.train(dec,criterion,Xt,Yt,Xv,Yv,1,64,tnet,0.1)


-- In[50]:

function rutil.check(y,Y)
    --print(y,Y)
    for i=1,y:nElement() do
        if(Y[i]==1) then return true end
        if(y[i]~=Y[i]) then return false end
    end
    return true
end
function rutil.kfacc2(outputs,targets)
    local Y,y = nil,nil;
    local N = outputs[1]:size(1)
    local C = outputs[1]:size(2)
    for k=1,--outputs do 
        Y = Y and torch.cat(Y,outputs[k]:reshape(N,1,C),2) or outputs[k]:reshape(N,1,C)
        y = y and torch.cat(y,targets[k]:reshape(N,1),2) or targets[k]:reshape(N,1)
    end
    local t,idx = Y:max(3)
    idx = idx:squeeze()
    local acc = 0
    for i=1,y:size(1) do
        if(rutil.check(idx[i],y[i])) then 
            acc = acc + 1
            --print('true')
        else
            --print('false')
        end
    end
    return acc
end


-- In[51]:

rutil.kvalid(dec,Xv,Yv,64,tnet)


-- In[52]:

torch.save('seq44.t7',dec)


-- In[ ]:



