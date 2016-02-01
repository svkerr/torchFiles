-- loads regression model and runs it
require 'nn'
require 'dpnn'

-- load module
module = torch.load("/home/skerr/torchFiles/regression_model.t7")

-- create dummy input
input = torch.Tensor(10,2):uniform(-1,1)

-- make predictions
for i=1, input:size(1) do
  pred = module:forward(input[i])
--  print(string.format('predicted output is: %f', pred))
  print(pred)
end

