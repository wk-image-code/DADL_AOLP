function [Preci,F1] = confusion(Gte,labely)
%{
Preci--precision;
F1--F1_score;

Gte--Test set real label
labely--Test set classification result label
%}
%% Construct confusion matrix
matrix = [];A=[];B=[];
num_lable = max(Gte);
for i=1:num_lable
    position = find(Gte==i);
    num_each_label = size(position,2);
    temp = labely(position);
    vector = get_vector(temp,num_lable);
    matrix=[matrix;vector];
    A(i,position) = Gte(position);
    B(i,position) = labely(position);
end
%% Precision accuracy, recall and F1-score were calculated according to confusion matrix.
Preci= 0;
Recall = 0;
F1 = 0;
for i=1:num_lable
    weight(i) = sum(matrix(i,:))/sum(matrix,'all');
    preci(i) = matrix(i,i)/sum(matrix(:,i));
    recall(i) = matrix(i,i)/sum(matrix(i,:));
    f1(i) = 2*preci(i)*recall(i)/(preci(i)+recall(i));
    if(isnan(preci(i)))
        preci(i)=0;
    end
    if(isnan(f1(i)))
        f1(i)=0;
    end
    Preci = Preci+preci(i)*weight(i);
    Recall = Recall + recall(i)*weight(i);
    F1 = F1 + f1(i)*weight(i);
end
Preci = Preci*100;
F1 = F1*100;
Acc_Te=sum((Gte-labely)==0)./length(labely);
fprintf('accuracy is %0.4f\n',Acc_Te*100);
fprintf('precision is %0.4f\n',Preci);
fprintf('recall is %0.4f\n',Recall*100);
fprintf('F1-score is %0.4f\n\n',F1);
%% plotconfusion
imagesc(matrix);
xlabel("Predicted class index");
ylabel("Actual class index");
set(gca,'fontsize',15.5);
colorbar;
% plotconfusion(A,B);
end
%% Gets each row vector of the confusion matrix
function [vector] = get_vector(temp,num_lable)
for i=1:num_lable
    each_num = find(temp==i);
    vector(i) = size(each_num,2);
end
end
