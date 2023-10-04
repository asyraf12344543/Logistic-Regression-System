clc; clear; close all;

if exist('splitmulticlassData.mat',"file")
    splitData = load("splitmulticlassData.mat");
    TRN = splitData.TRN;
    TST = splitData.TST;
else
    dataset2 = csvread('abalone.csv',1, 1); 
    
    % Separating data into different sets 
    [rows,cols] = size(dataset2) ;
    P = 0.80; 
    idx = randperm(rows);
    TRN = dataset2(idx(1:round(P*rows)),:) ; 
    TST = dataset2(idx(round(P*rows)+1:end),:) ;
    save('splitmulticlassData.mat','TRN','TST');

end

%split x and y
xtrn = TRN(:, 1:end-3);
xtst = TST(:, 1:end-3);

ytrn1 = TRN(:, end-2);
ytrn2 = TRN(:, end-1);
ytrn3 = TRN(:, end);
ytst1 = TST(:, end-2);
ytst2 = TST(:, end-1);
ytst3 = TST(:, end);

mtrn = length(ytrn1);
mtst = length(ytst1);

%normalized x
[xtrnNormalized, C, S] = normalize(xtrn);
xtrnNormalized = [ones(size(xtrnNormalized,1),1), xtrnNormalized];

xtstNormalized = normalize(xtst, "center", C, "scale", S);
xtstNormalized = [ones(size(xtstNormalized,1),1), xtstNormalized]; % add intercept term

%declare thetas
thetas = zeros(size(xtrnNormalized,2),1) + 0.05; % init all params to 0.05
initial_theta = thetas; 

%declare hyperparamaters
iterations = 5000;
alpha = 0.05;

thetas1 = initial_theta;
%class 1
for i = 1:iterations
    ttrxtrn = xtrnNormalized * thetas1;

    hipotesistrn1 = 1 ./ (1 + exp(-ttrxtrn));

    for j = 1:length(thetas1)
        gradientA(j,1) = 1/mtrn * (hipotesistrn1-ytrn1)' * xtrnNormalized(:,j);
        thetas1(j) =  thetas1(j) - alpha * gradientA(j,1);
    end

    costHistoryTrn1(i) = 1/mtrn * (-ytrn1' * log(hipotesistrn1) - (1-ytrn1)' * log(1-hipotesistrn1));
end

%class 2
thetas2 = initial_theta;

for i = 1:iterations
    ttrxtrn = xtrnNormalized * thetas2;

    hipotesistrn2 = 1 ./ (1 + exp(-ttrxtrn));

    for j = 1:length(thetas2)
        gradientA(j,1) = 1/mtrn * (hipotesistrn2-ytrn2)' * xtrnNormalized(:,j);
        thetas2(j) =  thetas2(j) - alpha * gradientA(j,1);
    end

    costHistoryTrn2(i) = 1/mtrn * (-ytrn2' * log(hipotesistrn2) - (1-ytrn2)' * log(1-hipotesistrn2));
end

%class 3
thetas3 = initial_theta;

for i = 1:iterations
    ttrxtrn = xtrnNormalized * thetas3;

    hipotesistrn3 = 1 ./ (1 + exp(-ttrxtrn));

    for j = 1:length(thetas3)
        gradientA(j,1) = 1/mtrn * (hipotesistrn3-ytrn3)' * xtrnNormalized(:,j);
        thetas3(j) =  thetas3(j) - alpha * gradientA(j,1);
    end

    costHistoryTrn3(i) = 1/mtrn * (-ytrn3' * log(hipotesistrn3) - (1-ytrn3)' * log(1-hipotesistrn3));
end

% Calculate the overall accuracy for each classifier
% Accuracy for class 1
predlabels1 = hipotesistrn1 >= 0.5; % Threshold at 0.5 
predlabeldouble1 = double (predlabels1);

figure;
ccclass1 = confusionchart(ytrn1, predlabeldouble1);
cmclass1 = confusionmat(ytrn1, predlabeldouble1);

% Calculate accuracy
accuracy1 = (cmclass1(2, 2) + cmclass1(1, 1)) / sum(cmclass1(:));

disp('Confusion Matrix for Class 1 (M):');
disp(cmclass1);
disp('Accuracy for Class 1 (M): ' + string(accuracy1));

% Accuracy for class 2
predlabels2 = hipotesistrn2 >= 0.5; % Threshold at 0.5 
predlabeldouble2 = double (predlabels2);

figure;
ccclass2 = confusionchart(ytrn2, predlabeldouble2);
cmclass2 = confusionmat(ytrn2, predlabeldouble2);

% Calculate accuracy
accuracy2 = (cmclass2(2, 2) + cmclass2(1, 1)) / sum(cmclass2(:));

disp('Confusion Matrix for Class 2 (F):');
disp(cmclass2);
disp('Accuracy for Class 2 (F): ' + string(accuracy2));

% Accuracy for class 3
predlabels3 = hipotesistrn3 >= 0.5; % Threshold at 0.5 
predlabeldouble3 = double (predlabels3);

figure;
ccclass3 = confusionchart(ytrn3, predlabeldouble3);
cmclass3 = confusionmat(ytrn3, predlabeldouble3);

% Calculate accuracy
accuracy3 = (cmclass3(2, 2) + cmclass3(1, 1)) / sum(cmclass3(:));

disp('Confusion Matrix for Class 3 (I):');
disp(cmclass3);
disp('Accuracy for Class 3 (I): ' + string(accuracy3));


% Input instance(s) for prediction
%numInstances = input("Enter the number of instances you want to input: ");

for k = 1:mtst
    input_instance = input("Please enter the instance number (from " + mtst + " records in the test set). Enter 0 tu exit: ");
    if(input_instance <= mtst && input_instance > 0)
        xinstance = xtstNormalized(input_instance, :);
        y1instance = ytst1(input_instance, :);
        y2instance = ytst2(input_instance, :);
        y3instance = ytst3(input_instance, :);

        ttrx1instance = xinstance * thetas1;
        hypothesisx1instance = 1 ./ (1 + exp(-ttrx1instance));

        ttrx2instance = xinstance * thetas2;
        hypothesisx2instance = 1 ./ (1 + exp(-ttrx2instance));


        ttrx3instance = xinstance * thetas3;
        hypothesisx3instance = 1 ./ (1 + exp(-ttrx3instance));

        disp(" Probability of Classifier 1 (M): " + hypothesisx1instance * 100 + "%");
        disp(" Probability of Classifier 2 (F): " + hypothesisx2instance * 100 + "%");
        disp(" Probability of Classifier 3 (I): " + hypothesisx3instance * 100 + "%");

        highest = 0;
        class = 0;
        className = "";
        actualClass = 0;
        actualClassName = "";
        if(hypothesisx1instance > hypothesisx2instance && hypothesisx1instance > hypothesisx3instance)
            highest = hypothesisx1instance;
            class = 1;
            className = "M";
        elseif(hypothesisx2instance > hypothesisx1instance && hypothesisx2instance > hypothesisx3instance)
            highest = hypothesisx2instance;
            class = 2;
            className = "F";

        else 
            highest = hypothesisx3instance;
            class = 3;
            className = "I";

        end

        if(y1instance == 1)
            actualClass = 1;
            actualClassName = "M";
        elseif(y2instance == 1)
            actualClass = 2;
            actualClassName = "F";
        else
            actualClass = 3;
            actualClassName = "I";
        end


        disp("Instance " +input_instance+ " belongs to the Class " +class+" (" +className+ ") with the highest probability value of " +highest*100+ "%");
        disp("Actual class for instance "+input_instance+" is Class "+actualClass+"("+actualClassName+")");
      
    elseif(input_instance > mtst)
        disp("Error: There are only "+mtst+ " records, Please enter a value between 1 and " +mtst);
    else
        break;
    end
end

%{                
    
    % Perform prediction on the input instance
    ttrinput = [1, (input_instance - C) ./ S];
    
    % Calculate the scores for each class
    scores1 = ttrinput * thetas1;
    scores2 = ttrinput * thetas2;
    scores3 = ttrinput * thetas3;
    
    % Perform classification based on the scores
    if scores1 >= scores2 && scores1 >= scores3
        predicted_class = 1;
    elseif scores2 >= scores1 && scores2 >= scores3
        predicted_class = 2;
    else
        predicted_class = 3;
    end
    
    % Display the predicted class label
    fprintf("Predicted class label for instance %d: %d\n", k, predicted_class);
end

%{


xvld = VLD(:,1:end-1);

yvld = VLD(:, end);

mvld = length(yvld);

xtst = TST(:,1:end-1); 
ytst = TST(:,end);
mtst = length(ytst);

[xtrnNormalized, C, S] = normalize(xtrn); % NOTE!!!! This one does not have the intercept term yet ya :) Don't forget to add the intercept term :)
xtrnNormalized = [ones(size(xtrnNormalized,1),1), xtrnNormalized]; % add intercept term

xvldNormalized = normalize(xvld, "center", C, "scale", S);
xvldNormalized = [ones(size(xvldNormalized,1),1), xvldNormalized]; % add intercept term

xtstNormalized = normalize(xtst, "center", C, "scale", S);
xtstNormalized = [ones(size(xtstNormalized,1),1), xtstNormalized]; % add intercept term

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
    tstttrx = xtstNormalized * thetas;

    hipotesistrn = 1 ./ (1 + exp(-trnttrx)); % hypotheses for ALL ROWS at once
    hipotesisvld = 1 ./ (1 + exp(-vldttrx)); % hypotheses for ALL ROWS at once
    hipotesistst = 1 ./ (1 + exp(-tstttrx)); % hypotheses for ALL ROWS at once

    for j = 1:length(thetas)
    gradientA(j,1) = 1/mtrn * (hipotesistrn-ytrn)' * xtrnNormalized(:,j);
    thetas(j) =  thetas(j) - alpha * gradientA(j,1);
    end

   

    %error = -ytrn' * log(hipotesistrn) - (1-ytrn)' * log(1-hipotesistrn);

    costHistorytrn(i) = 1/mtrn * ( -ytrn' * log(hipotesistrn) - (1-ytrn)' * log(1-hipotesistrn));
    costHistoryvld(i) = 1/mvld * ( -yvld' * log(hipotesisvld) - (1-yvld)' * log(1-hipotesisvld));
    costHistorytst(i) = 1/mtst * ( -ytst' * log(hipotesistst) - (1-ytst)' * log(1-hipotesistst));

    
   


end


predicted_labels = hipotesisvld >= 0.5; % Threshold at 0.5 for binary classification
predicted_labels = double(predicted_labels);
predicted_labelstst = hipotesistst >= 0.5;

cmtrn = confusionchart(yvld, predicted_labels);

% Create confusion matrix
confusion_matrix = zeros(2); % Assuming binary classification (0 and 1)
confusion_matrixtst = zeros(2);

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

% Count true positives, true negatives, false positives, false negatives
for i = 1:length(predicted_labelstst)
    if predicted_labelstst(i) == 1 && ytst(i) == 1
        confusion_matrixtst(1, 1) = confusion_matrixtst(1, 1) + 1; % True positive
    elseif predicted_labelstst(i) == 0 && ytst(i) == 0
        confusion_matrixtst(2, 2) = confusion_matrixtst(2, 2) + 1; % True negative
    elseif predicted_labelstst(i) == 1 && ytst(i) == 0
        confusion_matrixtst(1, 2) = confusion_matrixtst(1, 2) + 1; % False positive
    elseif predicted_labelstst(i) == 0 && ytst(i) == 1
        confusion_matrixtst(2, 1) = confusion_matrixtst(2, 1) + 1; % False negative
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


% Calculate accuracy
accuracytst = (confusion_matrixtst(1, 1) + confusion_matrixtst(2, 2)) / sum(confusion_matrixtst(:));

% Calculate precision
precisiontst = confusion_matrixtst(1, 1) / (confusion_matrixtst(1, 1) + confusion_matrixtst(1, 2));

% Calculate recall
recalltst = confusion_matrixtst(1, 1) / (confusion_matrixtst(1, 1) + confusion_matrixtst(2, 1));

% Calculate F-score
betatst = 1; % F1-score
fscoretst = (1 + betatst^2) * (precisiontst * recalltst) / ((betatst^2 * precisiontst) + recalltst);

% Calculate true positive rate (TPR)
TPRtst = confusion_matrixtst(1, 1) / (confusion_matrixtst(1, 1) + confusion_matrixtst(2, 1));

% Calculate false positive rate (FPR)
FPRtst = confusion_matrixtst(1, 2) / (confusion_matrixtst(1, 2) + confusion_matrixtst(2, 2));

% Display the results
disp('Confusion Matrix For Testing:');
disp(confusion_matrixtst);
disp('Accuracy For Testing: ' + string(accuracytst));
disp('Precision For Testing: ' + string(precisiontst));
disp('Recall For Testing: ' + string(recalltst));
disp('F-score For Testing: ' + string(fscoretst));
disp('True Positive Rate (TPR) For Testing: ' + string(TPRtst));
disp('False Positive Rate (FPR) For Testing: ' + string(FPRtst));





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
hold on
plot(1:iterations, costHistorytst, '-g');
hold off
title(['Training vs Validation vs Testing Cost  (' num2str(iterations) '-itrs at an alpha of ' num2str(alpha) ')']);
legend('Training Cost','Validation Cost', 'Testing Cost')

%}
%}