close all hidden
clear all
clc
%% Synthesize Data of Target Dynamics
x0 = [2; 0];
A = [-0.1 -1; 1 -0.1];
trueModel = @(t,y) A*y; %define the system model x_dot=Ax

numTimeSteps = 2000;
T = 15;
odeOptions = odeset(RelTol=1.e-7); %set the tolerance
t = linspace(0, T, numTimeSteps);%generate numTimeSteps points
[~, xTrain] = ode45(trueModel, t, x0, odeOptions);
xTrain = xTrain'; %transpose, xTrain is the true value of the system
%% Visualize the data
figure
plot(xTrain(1,:),xTrain(2,:),'LineWidth',2,'Color','blue')
title("Ground Truth Dynamics") 
xlabel("x(1)") 
ylabel("x(2)")
grid on
%% Define and Initialize Model Parameters
neuralOdeTimesteps = 40;
dt = t(2); %time step
timesteps = (0:neuralOdeTimesteps)*dt;% time series array
%initialize the parameters structure
neuralOdeParameters = struct;
%initialize the parameters for the fully connected operations in ODE models
stateSize = size(xTrain,1);
hiddenSize = 20; % 20 neurons
%define the params for fully connected layer1
neuralOdeParameters.fc1 = struct;
sz = [hiddenSize stateSize];
neuralOdeParameters.fc1.Weights = initializeGlorot(sz, hiddenSize, stateSize);
neuralOdeParameters.fc1.Bias = initializeZeros([hiddenSize 1]);
%define the params for fully connected layer2
neuralOdeParameters.fc2 = struct;
sz = [stateSize hiddenSize];
neuralOdeParameters.fc2.Weights = initializeGlorot(sz, stateSize, hiddenSize);
neuralOdeParameters.fc2.Bias = initializeZeros([stateSize 1]);
%Display the learnable params(weight and bias) of the model
neuralOdeParameters.fc1
neuralOdeParameters.fc2
%% training options
%options for Adam optimizer
gradDecay = 0.9;
sqGradDecay = 0.999;
learnRate = 0.002;
numIter = 1200;%maximum iterations
miniBatchSize = 200;
plotFrequency = 20; % every plotFrequency to solve the learned dynamics and display against groundTruth
%Initialize the averageGrad and averageSqGrad for Adam Solver
averageGrad = [];
averageSqGrad = [];
%Initialize the TrainingProgressMonitor object. 
% Because the timer starts when you create the monitor object, make sure that you create the object close to the training loop.
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Iteration","LearnRate"],XLabel="Iteration");
numTrainingTimesteps = numTimeSteps;
trainingTimesteps = 1:numTrainingTimesteps;
plottingTimesteps = 2:numTimeSteps;

iteration = 0;
%% Video Version
% 初始化 VideoWriter
videoFileName = 'TrainingProcess.mp4';
videoWriter = VideoWriter(videoFileName, 'Uncompressed AVI');
videoWriter.FrameRate = 5;
open(videoWriter);

iteration = 0;
while iteration < numIter && ~monitor.Stop
    iteration = iteration + 1;

    % 创建 mini-batch 数据
    [X, targets] = createMiniBatch(numTrainingTimesteps, neuralOdeTimesteps, miniBatchSize, xTrain);

    % 计算损失和梯度
    [loss, gradients] = dlfeval(@modelLoss, timesteps, X, neuralOdeParameters, targets);

    % 更新网络
    [neuralOdeParameters, averageGrad, averageSqGrad] = adamupdate(neuralOdeParameters, gradients, averageGrad, averageSqGrad, iteration, ...
        learnRate, gradDecay, sqGradDecay);

    % 记录损失
    recordMetrics(monitor, iteration, Loss=loss);

    % 绘制动态结果
    if mod(iteration, plotFrequency) == 0 || iteration == 1
        y = dlode45(@odeModel, t, dlarray(x0), neuralOdeParameters, DataFormat="CB");

        plot(xTrain(1, plottingTimesteps), xTrain(2, plottingTimesteps), "r--")
        hold on
        plot(y(1, :), y(2, :), "b-")
        hold off

        xlabel("x(1)")
        ylabel("x(2)")
        title("Predicted vs. Real Dynamics")
        legend("Training Ground Truth", "Predicted")
        drawnow

        % 保存帧到视频
        frame = getframe(gcf); % 捕获当前图像
        frame.cdata = imresize(frame.cdata, 2); % 将帧分辨率放大为原来的 2 倍
        repeatFactor = 2; % 每帧重复两次
        for r = 1:repeatFactor
          writeVideo(videoWriter, frame); % 写入视频文件
        end    
    end

    updateInfo(monitor, Iteration=iteration, LearnRate=learnRate);
    monitor.Progress = 100 * iteration / numIter;
end

% 关闭 VideoWriter
close(videoWriter);

% %training starts
% while iteration < numIter && ~monitor.Stop
%     iteration = iteration + 1;
% 
%     % Create mini-batch data
%     [X, targets] = createMiniBatch(numTrainingTimesteps, neuralOdeTimesteps, miniBatchSize, xTrain);
% 
%     % Evaluate network and compute loss and gradients
%     [loss,gradients] = dlfeval(@modelLoss,timesteps,X,neuralOdeParameters,targets);
% 
%     % Update network
%     [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iteration,...
%         learnRate,gradDecay,sqGradDecay);
% 
%     % Plot loss
%     recordMetrics(monitor,iteration,Loss=loss);
% 
%     % Plot predicted vs. real dynamics
%     if mod(iteration,plotFrequency) == 0  || iteration == 1
% 
%         % Use ode45 to compute the solution 
%         y = dlode45(@odeModel,t,dlarray(x0),neuralOdeParameters,DataFormat="CB");
% 
%         plot(xTrain(1,plottingTimesteps),xTrain(2,plottingTimesteps),"r--")
% 
%         hold on
%         plot(y(1,:),y(2,:),"b-")
%         hold off
% 
%         xlabel("x(1)")
%         ylabel("x(2)")
%         title("Predicted vs. Real Dynamics")
%         legend("Training Ground Truth", "Predicted")
% 
%         drawnow
%     end
%     updateInfo(monitor,Iteration=iteration,LearnRate=learnRate);
%     monitor.Progress = 100*iteration/numIter;
% end
load train; % 加载音频
sound(y, Fs); % 播放音频

