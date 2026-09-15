close all
clear all
clc
%set the random seed to 10
rng(10)
%% Synthesize Data of Target Dynamics
% % parameters of 1MG system
% Y = 0.61;
% E1_star = 1.05;
% P1_star = 0.3;
% Ma1 = 1.2;
% Da1 = 0.2;
% Mv1 = 12;
% Dv1 = 0.2;
% Q1_star = 0.571;
% delta_star = 0.347;
%% modified parameters for Simulink matching
Y = 0.61;
E1_star = 1.05;
P1_star = 0;
Ma1 = 1.2;
Da1 = 0.2;
Mv1 = 12;
Dv1 = 0.2;
Q1_star = 0;
delta_star = 0.347;
G=0.4528;
B=-0.415;
% system dynamics
% P=(E1_star + x(2))*(E1_star + x(2))*G - (E1_star + x(2))*(cos(x(1) + delta_star)*G+B*sin(x(1) + delta_star));
% Q=-(E1_star + x(2))*(E1_star + x(2))*B + (E1_star + x(2))*(cos(x(1) + delta_star)*B-G*sin(x(1) + delta_star));
f_value = @(t, x) [
    Da1 / Ma1 * (P1_star - ((E1_star + x(2))^2*G - (E1_star + x(2))*(cos(x(1) + delta_star)*G+B*sin(x(1) + delta_star)))) - x(1) / Ma1;
    Dv1 / Mv1 * (Q1_star - (-(E1_star + x(2))^2*B + (E1_star + x(2))*(cos(x(1) + delta_star)*B-G*sin(x(1) + delta_star)))) -  x(2) / Mv1
];
% %% time domain simulation
% %% 定义仿真时间和初始条件
% tspan = [0 100];         % 时间范围 0 到 10 秒
% x0 = [0; 0];            % 初始条件：[x1; x2]
% %% 使用 ode45 求解ODE
% [t, x] = ode45(f_value, tspan, x0);
% 
% %% 绘图显示 x1 和 x2 随时间的变化
% figure;
% 
% subplot(2,1,1);
% plot(t, x(:,1), 'LineWidth', 2);
% xlabel('Time (s)');
% ylabel('x_1');
% title('Time Domain Response of x_1');
% grid on;
% 
% subplot(2,1,2);
% plot(t, x(:,2), 'LineWidth', 2);
% xlabel('Time (s)');
% ylabel('x_2');
% title('Time Domain Response of x_2');
% grid on;
% % 初始条件
x0 = [-0.5; 1]; % 初始值 [delta, E]

% 时间设置
numTimeSteps = 2000;
T = 80;
t = linspace(0, T, numTimeSteps);

% 设置求解器选项
odeOptions = odeset('RelTol', 1.e-7); % 设置求解器的相对容差

% 求解系统
[~, xTrain] = ode45(f_value, t, x0, odeOptions);

% 转置结果以匹配期望的输出格式
xTrain = xTrain';

% Visualize
figure
plot(xTrain(1, :), xTrain(2, :), 'LineWidth', 2, 'Color', 'blue');
title("Ground Truth Dynamics");
xlabel("x(1)");
ylabel("x(2)");
xlim([-1,1])
ylim([-1,1])
grid on;
%% Define and Initialize Model Parameters
neuralOdeTimesteps = 40;
dt = t(2); %time step
timesteps = (0:neuralOdeTimesteps)*dt;% time series array
%initialize the parameters structure
neuralOdeParameters = struct;
%initialize the parameters for the fully connected operations in ODE models
stateSize = size(xTrain,1);
hiddenSize = 40; %original 20 neurons
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
learnRate = 0.002;% original value:0.002
numIter = 1194;%maximum iterations 1200
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
%training starts
while iteration < numIter && ~monitor.Stop
    iteration = iteration + 1;

    % Create mini-batch data
    [X, targets] = createMiniBatch(numTrainingTimesteps, neuralOdeTimesteps, miniBatchSize, xTrain);

    % Evaluate network and compute loss and gradients
    [loss,gradients] = dlfeval(@modelLoss,timesteps,X,neuralOdeParameters,targets);

    % Update network
    [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iteration,...
        learnRate,gradDecay,sqGradDecay);

    % Plot loss
    recordMetrics(monitor,iteration,Loss=loss);

    % Plot predicted vs. real dynamics
    if mod(iteration,plotFrequency) == 0  || iteration == 1

        % Use ode45 to compute the solution 
        y = dlode45(@odeModel,t,dlarray(x0),neuralOdeParameters,DataFormat="CB");

        plot(xTrain(1,plottingTimesteps),xTrain(2,plottingTimesteps),"r--")

        hold on
        plot(y(1,:),y(2,:),"b-")
        hold off

        xlabel("x(1)")
        ylabel("x(2)")
        title("Predicted vs. Real Dynamics")
        legend("Training Ground Truth", "Predicted")

        drawnow
    end
    updateInfo(monitor,Iteration=iteration,LearnRate=learnRate);
    monitor.Progress = 100*iteration/numIter;
end
% %% multiple trajectories joint optimization
% %% 多条轨迹的初始条件
% x0Set = [
%     -0.5, 1;
%     0.25,0.75;
%     0,0.5;
%     0.25,-0.5
% ]; % 添加更多初始条件
% 
% numTrajectories = size(x0Set, 1); % 根据初始条件数量决定轨迹数
% xTrainSet = cell(numTrajectories, 1); % 存储每条轨迹的数据
% 
% % 生成多条轨迹
% for i = 1:numTrajectories
%     x0 = x0Set(i, :)'; % 当前轨迹的初始条件
%     [~, xTrain] = ode45(f_value, t, x0, odeOptions);
%     xTrainSet{i} = xTrain'; % 转置为列向量形式
% end
% 
% %% 初始化训练参数
% iteration = 0;
% totalGradients = []; % 初始化总梯度
% 
% % Adam 优化器参数
% averageGrad = [];
% averageSqGrad = [];
% learnRate = 0.001/2;
% gradDecay = 0.9;
% sqGradDecay = 0.999;
% 
% %% 训练过程
% while iteration < numIter && ~monitor.Stop
%     iteration = iteration + 1;
% 
%     totalLoss = 0; % 初始化总损失
% 
%     % 遍历所有轨迹进行联合优化
%     for j = 1:numTrajectories
%         xTrain = xTrainSet{j}; % 当前轨迹数据
% 
%         % 创建 mini-batch 数据
%         [X, targets] = createMiniBatch(numTrainingTimesteps, neuralOdeTimesteps, miniBatchSize, xTrain);
% 
%         % 计算当前轨迹的损失和梯度
%         [loss, gradients] = dlfeval(@modelLoss, timesteps, X, neuralOdeParameters, targets);
% 
%         % 累加当前轨迹的损失
%         totalLoss = totalLoss + loss;
% 
%         % 累加梯度
%         if isempty(totalGradients)
%             totalGradients = gradients;  % 第一次直接赋值
%         else
%             % 遍历每个字段（如 fc1 和 fc2）
%             gradientFields = fieldnames(gradients);
%             for k = 1:numel(gradientFields)
%                 field = gradientFields{k};
%                 subFields = fieldnames(gradients.(field));
%                 for subFieldIdx = 1:numel(subFields)
%                     subField = subFields{subFieldIdx};
%                     totalGradients.(field).(subField) = ...
%                         totalGradients.(field).(subField) + gradients.(field).(subField);
%                 end
%             end
%         end
%     end
% 
%     % 计算总损失的平均值
%     totalLoss = totalLoss / numTrajectories;
% 
%     % 计算平均梯度
%     gradientFields = fieldnames(totalGradients);
%     for k = 1:numel(gradientFields)
%         field = gradientFields{k};
%         subFields = fieldnames(totalGradients.(field));
%         for subFieldIdx = 1:numel(subFields)
%             subField = subFields{subFieldIdx};
%             totalGradients.(field).(subField) = ...
%                 totalGradients.(field).(subField) / numTrajectories;
%         end
%     end
% 
%     % 更新模型参数
%     [neuralOdeParameters, averageGrad, averageSqGrad] = adamupdate(neuralOdeParameters, totalGradients, averageGrad, averageSqGrad, iteration, ...
%         learnRate, gradDecay, sqGradDecay);
% 
%     % 记录训练过程中的损失
%     recordMetrics(monitor, iteration, Loss=totalLoss);
% 
%     % 更新监视器
%     updateInfo(monitor, Iteration=iteration, LearnRate=learnRate);
%     monitor.Progress = 100 * iteration / numIter;
% end

%% 训练完成
disp('Training completed.');