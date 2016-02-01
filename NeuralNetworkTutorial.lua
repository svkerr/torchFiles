-- to get familiar with the 'dp' deep learning library designed for streamline R&D using Torch7

--[[required packages]]--
require 'dp'
require 'optim'
require 'cutorch'
require 'cunnx'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th NeuralNetworkTutorial.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('accUpdate', false, 'change how gradParameters are used and updated')
cmd:option('learningRate', 0.1, 'learning rate at t=0')
cmd:option('lrDecay', 'linear', 'type of learning rate decay: adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--hiddenSize', '{200,200}', 'number of hidden units per layer')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
  table.print(opt)
end

--[[preprocessing]]--
-- the -zca and --standardize cmd-line arguments can be toggled on to perform some data preprocessing
local input_preprocess = {}
if opt.standardize then
  table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
  table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
  table.insert(input_preprocess, dp.GCN())
  table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--[[data]]--
--[[We intend to build and train a neural network so we need some data, which we encapsulate in a DataSource object. 
dp provides the option of training on different datasets, notably MNIST, NotMNIST, CIFAR-10 or CIFAR-100. 
The default is MNIST.  However, you can use the --dataset argument to specify a different image classification dataset.--]]

if opt.dataset == 'Mnist' then
  ds = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   ds = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   ds = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   ds = dp.Cifar100{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end

--[[model]]--
--[[ Ok so we have a DataSource, now we need a model. Let's build a multi-layer perceptron (MLP) with one or more parameterized non-linear layers--]]

model = nn.Sequential()
model:add(nn.Convert(ds:ioShapes(), 'bf')) -- flatten images to batchsize x nFeature

--hidden layers--
inputSize = ds:featureSize()                    -- for Mnist, ds:featureSize()==784 = 28x28x1 (hxwxc)
for i, hiddenSize in ipairs(opt.hiddenSize) do  -- for Mnist, this creates two composite layers:(linear, tanh, dropout)x2
  model:add(nn.Linear(inputSize, hiddenSize))   -- parameters to be learned 
  if opt.batchNorm then
    model:add(nn.BatchNormalization(hiddensize))
  end
  model:add(nn.Tanh())
  if opt.dropout then
    model:add(nn.Dropout())
  end
  inputSize = hiddenSize
end

--output layer--
model:add(nn.Linear(inputSize, #(ds:classes())))
model:add(nn.LogSoftMax())

--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
  ad = dp.AdaptiveDecay{max_wait=opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
  opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
  acc_update = opt.accUpdate,
  loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),  --ModuleCriterion() is in dpnn
  epoch_callback = function(model, report)                               --called every epoch,typically for lr decay and such
    -- learning rate decay
    if report.epoch > 0 then
      if opt.lrDecay == 'adaptive' then
        opt.learningRate = opt.learningRate * ad.decay
        ad.decay = 1
      elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
        opt.learningRate = opt.schedule[report.epoch]
      elseif opt.lrDecay == 'linear' then
        opt.learningRate = opt.learningRate + opt.decayFactor
      end
    end
  end,
  callback = function(model, report)    -- called every batch, update the model parameters, gather statistics decay lr etc.,
    if opt.accUpdate then
      model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
    else
      model:updateGradParameters(opt.momentum, nil, true)    -- affects gradParams, 'true' flag uses Nesterov
--      model:updateGradParameters(opt.momentum)               -- affects gradParams, not use Nesterov 
      model:updateParameters(opt.learningRate)    -- affects params
    end
    model:maxParamNorm(opt.maxOutNorm)            -- affects params
    model:zeroGradParameters()                    -- affects gradParams
  end,
  feedback = dp.Confusion(),
  sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
  progress = opt.progress 
}

valid = dp.Evaluator{
  feedback = dp.Confusion(),
  sampler = dp.Sampler{batch_size = opt.batchSize}
}

test = dp.Evaluator{
  feedback = dp.Confusion(),
  sampler = dp.Sampler{batch_size = opt.batchSize}
}
--[[Experiment]]--

xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}
xp:run(ds)

--[[
When loading the saved experiment per the dp tutorial
instead of require 'cuda' use the following instead:
require 'optim'
require 'cutorch'
--]]
 
