load('lidar_xy.mat')
plot(points(:,1),points(:,2),'o');
hold on


sampleSize = 3; % number of points to sample per trial
maxDistance = 0.01;
fitLineFcn = @(points) [polyfit(points(1:2,1),points(1:2,2),1), polyfit(points(1:2,1),points(1:2,2),1)-[0 points(3,2)-points(1,2)]]; % fit function using polyfit
evalLineFcn = ...   % distance evaluation function
      @(model, points) min(sum((points(:, 2) - polyval(model(1:2), points(:,1))).^2,2), sum((points(:, 2) - polyval(model(3:4), points(:,1))).^2,2));
[modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
      sampleSize,maxDistance, 'MaxNumTrials', 1000000, 'Confidence', 99.999999);
x1 = min(points(:, 1)); x2 =  max(points(:, 1));
y1 = polyval(modelRANSAC(1:2), x1);
y2 = polyval(modelRANSAC(1:2), x2);
plot([x1 x2],[y1 y2]);

y1s = polyval(modelRANSAC(3:4), x1);
y2s = polyval(modelRANSAC(3:4), x2);
plot([x1 x2],[y1s y2s]);

  % 
% 
% [x1 y1 x2 y2] = get_lines(points)
% 
% plot(x1, y1, 'g-')
% plot(x2, y2, 'g-')
% legend('Noisy points','Least squares fit','Robust fit');
% hold off
% 
% function [x1 y1 x2 y2] = get_lines(points)
%     sampleSize = 4; % number of points to sample per trial
%     maxDistance = 0.2; % max allowable distance for inliers
% 
%     fitLineFcn = @(points) [polyfit(points(1:2,1),points(1:2,2),1), polyfit(points(1:2,1),points(1:2,2),1)-[0 points(2,2)-points(1,2)]]; % fit function using polyfit
%     evalLineFcn = ...   % distance evaluation function
%       @(model, points) sum((points(:, 2) - polyval(model(1:2), points(:,1))).^2,2) + sum((points(:, 2) - polyval(model(3:4), points(:,1))).^2,2);
% 
%     [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
%       sampleSize,maxDistance);
%     modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
%     inlierPts = points(inlierIdx,:);
%     x1 = [min(inlierPts(:,1)) max(inlierPts(:,1))];
%     y1 = modelInliers(1)*x1 + modelInliers(2);
%     points = points(~inlierIdx,:);
%     
%     [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
%       sampleSize,maxDistance);
%     modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
%     inlierPts = points(inlierIdx,:);
%     x2 = [min(inlierPts(:,1)) max(inlierPts(:,1))];
%     y2 = modelInliers(1)*x2 + modelInliers(2);
% end