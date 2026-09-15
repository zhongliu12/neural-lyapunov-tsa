function plotTrueAndPredictedSolutions(xTrue,xPred)

xPred = squeeze(xPred)';

err = mean(abs(xTrue(2:end,:) - xPred), "all");

plot(xTrue(:,1),xTrue(:,2),'color',[0.2 0.5 0.9 0.4],'LineWidth',2.5)
hold
plot(xPred(:,1),xPred(:,2),'color',[0.2 0.5 0.9 1],'LineWidth',2.5,'LineStyle','--')
hold off
title("Absolute Error = " + num2str(err,"%.4f"))
xlabel("x(1)")
ylabel("x(2)")

% xlim([-1 1])
% ylim([-1 1])
grid
legend("Ground Truth","Predicted",'Location','southeast')

end
function [outputArg1,outputArg2] = untitled6(inputArg1,inputArg2)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end