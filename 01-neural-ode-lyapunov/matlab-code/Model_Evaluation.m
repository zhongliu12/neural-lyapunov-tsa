close all
% total timespan
tPred = t;
% initial values
x0Pred1 = sqrt([2;2]);
x0Pred2 = [-1;2];
x0Pred3 = [0;2];
x0Pred4 = [-2;0];
% solve for the groundTruth trajectories
[~, xTrue1] = ode45(trueModel, tPred, x0Pred1, odeOptions);
[~, xTrue2] = ode45(trueModel, tPred, x0Pred2, odeOptions);
[~, xTrue3] = ode45(trueModel, tPred, x0Pred3, odeOptions);
[~, xTrue4] = ode45(trueModel, tPred, x0Pred4, odeOptions);
% solve for the predicted trajectories by Neural ODEs
xPred1 = dlode45(@odeModel,tPred,dlarray(x0Pred1),neuralOdeParameters,DataFormat="CB");
xPred2 = dlode45(@odeModel,tPred,dlarray(x0Pred2),neuralOdeParameters,DataFormat="CB");
xPred3 = dlode45(@odeModel,tPred,dlarray(x0Pred3),neuralOdeParameters,DataFormat="CB");
xPred4 = dlode45(@odeModel,tPred,dlarray(x0Pred4),neuralOdeParameters,DataFormat="CB");
%plot
figure
subplot(2,2,1)
plotTrueAndPredictedSolutions(xTrue1, xPred1);
subplot(2,2,2)
plotTrueAndPredictedSolutions(xTrue2, xPred2);
subplot(2,2,3)
plotTrueAndPredictedSolutions(xTrue3, xPred3);
subplot(2,2,4)
plotTrueAndPredictedSolutions(xTrue4, xPred4);