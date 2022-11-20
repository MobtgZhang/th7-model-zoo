require('torch')
require('cunn')
require('nn')
require('optim')
require('paths')
require('cutorch')
mnist = require('mnist')
weight_init = require('weight-init')
function prepare_dataset()
  fullset = mnist.traindataset()
  test_dataset = mnist.testdataset()
  print('Split validation set')
  train_dataset = {
      size = 50000,
      data = fullset.data[{{1,50000}}]:double(),
      label = fullset.label[{{1,50000}}]
  }
  validate_dataset = {
      size = 10000,
      data = fullset.data[{{50001,60000}}]:double(),
      label = fullset.label[{{50001,60000}}]
  }
  return train_dataset,validate_dataset,test_dataset
end
function build_model(weight_init)
  model = nn.Sequential()
  model:add(nn.Reshape(1,28,28))
  model:add(nn.MulConstant(1/256.0*3.2))
  model:add(nn.SpatialConvolutionMM(1,20,5,5,1,1,0,0))
  model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))
  model:add(nn.SpatialConvolutionMM(20, 50, 5, 5, 1, 1, 0, 0))
  model:add(nn.SpatialMaxPooling(2, 2 , 2, 2, 0, 0))
  model:add(nn.Reshape(4*4*50))
  model:add(nn.Linear(4*4*50, 500))
  model:add(nn.ReLU())
  model:add(nn.Linear(500, 10))
  model:add(nn.LogSoftMax())
  if weight_init then
    model = weight_init(model,'xavier')
  end
  return model
end
function main()
  -- preparing dataset
  print("preparing dataset")
  train_dataset,validate_dataset,test_dataset = prepare_dataset()
  print("Normalize the dataset")
  train_dataset.data = train_dataset.data - train_dataset.data:mean()
  validate_dataset.data = validate_dataset.data - validate_dataset.data:mean()
  print('train set information')
  print(train_dataset)
  print('validation set information')
  print(validate_dataset)
  print('test set information')
  print(test_dataset)
  -- preparing model
  model = build_model(weight_init)
  criterion = nn.ClassNLLCriterion()
  model = model:cuda()
  print("model information")
  print(model)
  logdir = 'logs'
  if not paths.dir(logdir) then
    paths.mkdir(logdir)
  end
  modeldir = logdir .. '/save.pt'
  criterion = criterion:cuda()
  train_dataset.data = train_dataset.data:cuda()
  validate_dataset.data = validate_dataset.data:cuda()
  test_dataset.data = test_dataset.data:cuda()
  sgd_params = {
    learningRate = 1e-2,
    learningRateDecay = 1e-4,
    weightDecay = 1e-3,
    momentum = 1e-4
  }
  params,gradient = model:getParameters()
  -- every step of the training process
  step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(train_dataset.size)
    batch_size = batch_size or 200
    for t = 1,train_dataset.size,batch_size do
      local size = math.min(t+batch_size-1,train_dataset.size) - t
      local inputs = torch.Tensor(size,28,28):cuda()
      local targets = torch.Tensor(size):cuda()
      for i =1,size do
        local input = train_dataset.data[shuffle[i+t]]
        local target = train_dataset.label[shuffle[i+t]]
        inputs[i] = input
        targets[i] = target
      end
      targets:add(1) -- notice that torch library tensor index start with 1
      local feval = function (params_new)
        if params ~= params_new then
          params:copy(params_new)
        end
        gradient:zero()
        local loss = criterion:forward(model:forward(inputs),targets)
        model:backward(inputs,criterion:backward(model.output,targets))
        return loss,gradient
      end
      _,fs = optim.sgd(feval,params,sgd_params)
      count = count + 1
      current_loss = current_loss + fs[1] 
    end
    return current_loss/count
  end
  eval = function(dataset,batch_size)
    local count = 0
    batch_size = batch_size or 200
    for i = 1,dataset.size,batch_size do
      local size = math.min(i+batch_size-1,dataset.size) - i
      local inputs = dataset.data[{{i,i+size-1}}]:cuda()
      local targets = dataset.label[{{i,i+size-1}}]:cuda()
      local outputs = model:forward(inputs)
      local _,indices = torch.max(outputs,2)
      indices:add(-1)
      indices = indices:cuda()
      local guessed_right = indices:eq(targets):sum()
      count = count + guessed_right
    end
    return count/dataset.size
  end
  max_iters = 30
  print("Start training")
  do 
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1
    for k=1,max_iters do
      local loss = step()
      print(string.format('Eopch:%d Current loss: %0.4f',k,loss))
      local accuracy = eval(validate_dataset)
      print(string.format('Accuracy on the validate dataset :%0.2f%%',accuracy*100))
      if accuracy<last_accuracy then
        if decreasing >threshold then
          decreasing = decreasing +1
        else
          decreasing = 0
        end
        last_accuracy = accuracy
      end
    end
  end
  print('Saving model ...')
  torch.save(modeldir,model)
  test_dataset.data = test_dataset.data:double()
  local accuracy = eval(test_dataset)
  print(string.format('Accuracy on the test set: %0.2f%%',accuracy*100))
end
main()


