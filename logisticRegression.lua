-- A binary logistic regressor module with 2 inputs and 1 output

require 'nn'
require 'dpnn'

-- create model
module = nn.Sequential()
module:add(nn.Linear(2,1))
module:add(nn.Sigmoid())

-- establish criterion: since input is binary we use
criterion = nn.BCECriterion()

-- create dummy data set
inputs = torch.Tensor(10,2):uniform(-1,1)
targets = torch.Tensor(10):random(0,1)

-- function for one epoch of stochastic gradient descent (SGD)
function trainEpoch(module, criterion, inputs, targets)
  for i=1, inputs:size(1) do
    idx = math.random(1, inputs:size(1))
    input, target = inputs[idx], targets:narrow(1, idx, 1) -- can't use targets[idx] due to resulting shape
    -- forward
    output = module:forward(input)
    loss = criterion:forward(output, target)
    -- backward
    gradOutput = criterion:backward(output, target)
    module:zeroGradParameters()
    gradInput = module:backward(input, gradOutput)
    -- update
    module:updateGradParameters(0.9)  -- update with momentum (dpnn)
    module:updateParameters(0.1)      -- W = W - 0.1*dL/dW
  end
end

-- Do 100 epochs to train the module:
for i=1,100 do
  trainEpoch(module, criterion, inputs, targets)
  print(string.format('Epoch %d ; BCE err = %f', i, loss))
end

-- Save, Load and Test the trained module:
-- Equivalent to .caffemodel using Torch's save() serializer
torch.save("/home/skerr/torchFiles/regression_model.t7", module)
x = torch.Tensor(1,2):uniform(-1,1)
print(module:forward(x))
