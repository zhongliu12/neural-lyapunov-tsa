function xDot = computeDerivative(x, neuralOdeParameters)
    % 计算神经网络的输出（导数值）
    % x: 当前状态，列向量
    % neuralOdeParameters: 模型的参数，包括 fc1 和 fc2

    % fc1: 第一个全连接层
    fc1Weights = neuralOdeParameters.fc1.Weights;
    fc1Bias = neuralOdeParameters.fc1.Bias;

    % fc2: 第二个全连接层
    fc2Weights = neuralOdeParameters.fc2.Weights;
    fc2Bias = neuralOdeParameters.fc2.Bias;

    % 计算第一层输出
    h = tanh(fc1Weights* x + fc1Bias);

    % 计算第二层输出
    xDot = fc2Weights * h + fc2Bias;
end
