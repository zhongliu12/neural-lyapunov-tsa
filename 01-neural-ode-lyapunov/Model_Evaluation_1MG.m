clc
close all
% load('neural_ode_weights_1MG.mat')
% neuralOdeParameters = struct;
% neuralOdeParameters.fc1.Weights=dlarray(W1);
% neuralOdeParameters.fc1.Bias=dlarray(b1);
% neuralOdeParameters.fc2.Weights=dlarray(W2);
% neuralOdeParameters.fc2.Bias=dlarray(b2);
% total timespan
numTimeSteps = 2000;
T = 80;
t = linspace(0, T, numTimeSteps);
tPred = t;
% initial values
x0Pred1 = [-0.5;1];
x0Pred2 = [0.25;0.75];
x0Pred3 = [0;0.5];
x0Pred4 = [0.25;-0.5];
% parameters of 1MG system
Y = 0.61;
E1_star = 1.05;
P1_star = 0;
Ma1 = 1.2;
Da1 = 0.2;
Mv1 = 12;
Dv1 = 0.2;
Q1_star = 0;
delta_star = 0.347;
% system dynamics
f_value = @(t, x) [
    Da1 / Ma1 * (P1_star - ((E1_star + x(2))^2*G - (E1_star + x(2))*(cos(x(1) + delta_star)*G+B*sin(x(1) + delta_star)))) - x(1) / Ma1;
    Dv1 / Mv1 * (Q1_star - (-(E1_star + x(2))^2*B + (E1_star + x(2))*(cos(x(1) + delta_star)*B-G*sin(x(1) + delta_star)))) -  x(2) / Mv1
];
% 设置求解器选项
odeOptions = odeset('RelTol', 1.e-7); % 设置求解器的相对容差
% solve for the groundTruth trajectories
[~, xTrue1] = ode45(f_value, tPred, x0Pred1, odeOptions);
[~, xTrue2] = ode45(f_value, tPred, x0Pred2, odeOptions);
[~, xTrue3] = ode45(f_value, tPred, x0Pred3, odeOptions);
[~, xTrue4] = ode45(f_value, tPred, x0Pred4, odeOptions);
% solve for the predicted trajectories by Neural ODEs
xPred1 = dlode45(@odeModel,tPred,dlarray(x0Pred1),neuralOdeParameters,DataFormat="CB");
xPred2 = dlode45(@odeModel,tPred,dlarray(x0Pred2),neuralOdeParameters,DataFormat="CB");
xPred3 = dlode45(@odeModel,tPred,dlarray(x0Pred3),neuralOdeParameters,DataFormat="CB");
xPred4 = dlode45(@odeModel,tPred,dlarray(x0Pred4),neuralOdeParameters,DataFormat="CB");
%plot
figure(1)
subplot(2,2,1)
plotTrueAndPredictedSolutions(xTrue1, xPred1);
subplot(2,2,2)
plotTrueAndPredictedSolutions(xTrue2, xPred2);
subplot(2,2,3)
plotTrueAndPredictedSolutions(xTrue3, xPred3);
xlim([-0.1,0])
subplot(2,2,4)
plotTrueAndPredictedSolutions(xTrue4, xPred4);
% syms x1 x2
% x1_plot=-1:0.1:1;
% x2_plot=-1:0.1:1;
% % x_dot from actual system
% f_actual1_plot =[
%     Da1 / Ma1 .* (P1_star - Y .* (E1_star + x2_plot) .* cos((x1_plot + delta_star) + 0.74)) - x1_plot / Ma1;
%     Dv1 / Mv1 .* (Q1_star - Y .* (E1_star + x2_plot) .* sin((x1_plot + delta_star) + 0.74)) -  x1_plot / Mv1
% ];
% %x_dot from neural network
% f_neural1_plot=computeDerivative([x1_plot;x2_plot], neuralOdeParameters);
% figure(2)
% plot(x1_plot,f_actual1_plot(1,:),'-b',x1_plot,f_neural1_plot(1,:))
% legend('Actual','NN')
% xlabel('x1')
% ylabel('x1_(dot)')
% err = mean(abs(f_actual1_plot(1,:) - f_neural1_plot(1,:)), "all");
% title("Absolute Error = " + num2str(err,"%.4f"))
% figure(3)
% plot(x2_plot,f_actual1_plot(2,:),'-b',x2_plot,f_neural1_plot(2,:))
% legend('Actual','NN')
% err = mean(abs(f_actual1_plot(2,:) - f_neural1_plot(2,:)), "all");
% title("Absolute Error = " + num2str(err,"%.4f"))
% xlabel('x2')
% ylabel('x2_(dot)')
figure(3)
plotTrueAndPredictedSolutions(xTrue1, xPred1);