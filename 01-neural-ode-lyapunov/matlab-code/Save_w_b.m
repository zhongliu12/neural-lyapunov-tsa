% 假设你已经有 weights 和 biases
% weights 和 biases 是 dlarray 格式

% 提取参数
W1 = extractdata(neuralOdeParameters.fc1.Weights); % 输入到隐藏层的权重
b1 = extractdata(neuralOdeParameters.fc1.Bias);  % 输入到隐藏层的偏置
W2 = extractdata(neuralOdeParameters.fc2.Weights); % 隐藏层到输出的权重
b2 = extractdata(neuralOdeParameters.fc2.Bias);  % 隐藏层到输出的偏置

% 保存为 .mat 文件
save('neural_ode_weights_3MG_newjoint.mat', 'W1', 'b1', 'W2', 'b2');
