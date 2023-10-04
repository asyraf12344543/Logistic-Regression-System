clc; clear; close all;

if exist('splitData.mat',"file")
    splitData = load("splitData.mat");
    TRN = splitData.TRN;
    VLD = splitData.VLD;
    TST = splitData.TST;
else
    dataset = csvread('cardio.csv',1, 0); % just a sample binary classification dataset
    
    % Separating data into different sets 
    [rows,cols] = size(dataset) ;
    P = 0.80; % 80% goes into setA and the remainder into... well, remainder :)
    idx = randperm(rows);
    TRN_VLD = dataset(idx(1:round(P*rows)),:) ; 
    TST = dataset(idx(round(P*rows)+1:end),:) ;
    [rows2,cols2] = size(TRN_VLD) ;
    P2 = 0.35;
    TRN = TRN_VLD(1:round(P2*rows2),:);
    VLD = TRN_VLD(round(P2*rows2)+1:end,:);
    save('splitData.mat','TRN','VLD','TST');
end

% Example normalization code
xtrn = TRN(:,1:end-1); % This line extracts the input features from the setA dataset. It selects all rows and all columns except for the last column (end-1), assuming that the last column represents the target variable.
xvld = VLD(:,1:end-1);

ytrn = TRN(:,end);
yvld = VLD(:, end);

mtrn = length(ytrn);
mvld = length(yvld);

[xtrnNormalized, C, S] = normalize(xtrn); % NOTE!!!! This one does not have the intercept term yet ya :) Don't forget to add the intercept term :)
xtrnNormalized = [ones(size(xtrnNormalized,1),1), xtrnNormalized]; % add intercept term

xvldNormalized = normalize(xvld, "center", C, "scale", S);
xvldNormalized = [ones(size(xvldNormalized,1),1), xvldNormalized]; % add intercept term

% to normalize another dataset from the same distribution (or same 'mother'
% dataset...) - xAnotherSetNormalized = normalize(xAnotherSet, "center", C, "scale", S);

thetas = zeros(size(xtrnNormalized,2),1) + 0.05; % init all params to 0.05
initial_theta = thetas; % will be used later...

% Theta transpose x :)


iterations = 50000;
alpha = 1.05;
for i = 1:iterations
    trnttrx = xtrnNormalized * thetas;
    vldttrx = xvldNormalized * thetas;
    hipotesistrn = 1 ./ (1 + exp(-trnttrx)); % hypotheses for ALL ROWS at once
    hipotesisvld = 1 ./ (1 + exp(-vldttrx)); % hypotheses for ALL ROWS at once

    for j = 1:length(thetas)
    gradientA(j,1) = 1/mtrn * (hipotesistrn-ytrn)' * xtrnNormalized(:,j);
    thetas(j) =  thetas(j) - alpha * gradientA(j,1);
    end

   

    %error = -ytrn' * log(hipotesistrn) - (1-ytrn)' * log(1-hipotesistrn);

    costHistorytrn(i) = 1/mtrn * ( -ytrn' * log(hipotesistrn) - (1-ytrn)' * log(1-hipotesistrn));
    costHistoryvld(i) = 1/mvld * ( -yvld' * log(hipotesisvld) - (1-yvld)' * log(1-hipotesisvld));

    
   


end


predicted_labels = hipotesisvld >= 0.5; % Threshold at 0.5 for binary classification

% Create confusion matrix
confusion_matrix = zeros(2); % Assuming binary classification (0 and 1)

% Count true positives, true negatives, false positives, false negatives
for i = 1:length(predicted_labels)
    if predicted_labels(i) == 1 && yvld(i) == 1
        confusion_matrix(1, 1) = confusion_matrix(1, 1) + 1; % True positive
    elseif predicted_labels(i) == 0 && yvld(i) == 0
        confusion_matrix(2, 2) = confusion_matrix(2, 2) + 1; % True negative
    elseif predicted_labels(i) == 1 && yvld(i) == 0
        confusion_matrix(1, 2) = confusion_matrix(1, 2) + 1; % False positive
    elseif predicted_labels(i) == 0 && yvld(i) == 1
        confusion_matrix(2, 1) = confusion_matrix(2, 1) + 1; % False negative
    end
end

% Calculate accuracy
accuracy = (confusion_matrix(1, 1) + confusion_matrix(2, 2)) / sum(confusion_matrix(:));

% Calculate precision
precision = confusion_matrix(1, 1) / (confusion_matrix(1, 1) + confusion_matrix(1, 2));

% Calculate recall
recall = confusion_matrix(1, 1) / (confusion_matrix(1, 1) + confusion_matrix(2, 1));

% Calculate F-score
beta = 1; % F1-score
fscore = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall);

% Calculate true positive rate (TPR)
TPR = confusion_matrix(1, 1) / (confusion_matrix(1, 1) + confusion_matrix(2, 1));

% Calculate false positive rate (FPR)
FPR = confusion_matrix(1, 2) / (confusion_matrix(1, 2) + confusion_matrix(2, 2));

% Display the results
disp('Confusion Matrix:');
disp(confusion_matrix);
disp('Accuracy: ' + string(accuracy));
disp('Precision: ' + string(precision));
disp('Recall: ' + string(recall));
disp('F-score: ' + string(fscore));
disp('True Positive Rate (TPR): ' + string(TPR));
disp('False Positive Rate (FPR): ' + string(FPR));







% from the hipotesis... you can get the error :) That one how to do?
% You can get the error all at once by doing a vector-vector subtraction :)

% Gradient Descent - update of parameters code
%for j = 1:length(thetas)
%    gradientA(j,1) = 1/m * (theErrorVector)' * xNormalized(:,j);
%    thetas(j) =  thetas(j) - alpha * gradientA(j,1);
%end

% COST code (in vector format, so we know the cost at each epoch/iteration
% the variable 'i' of course indicates that this code belongs in a loop :)


% Plotting. 'hold on' ensures the current figure is "held" so that whatever
% you plot next, will be ON it. If you don't hold on, a new figure will be
% created... we don't want that. We want to see both costs for training and
% validation :)
figure;
plot(1:iterations, costHistorytrn, '--b');
hold on
plot(1:iterations, costHistoryvld, '-r');
hold off
title(['Training vs Validation Costs (' num2str(iterations) '-itrs at an alpha of ' num2str(alpha) ')']);
legend('Training Cost','Validation Cost')
%}


