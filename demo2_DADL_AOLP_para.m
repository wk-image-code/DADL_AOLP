% Demo for DADL-AOLP

% In this example, lambda1 and lambda2 are adjusted to see the effect between them on the result
% You can adjust the number of atoms and other hyperparameters to see their effect on the results

clear;close all;
load('./data/scene15');
%% Hyper-parameters
lambda1_=[1e-5 1e-4 1e-3 1e-2 1e-1 1];
lambda2_=[1e-5 1e-4 1e-3 1e-2 1e-1 1];
lambda3_=[1e-4];
lambda4_=[1e-1];
mu_=[1];
anum_=[50];
maxIter=25;
for a=1:length(anum_)
    for a1=1:length(lambda1_)
        for a2=1:length(lambda2_)
            for a3=1:length(lambda3_)
                for a4=1:length(lambda4_)
                    for a5=1:length(mu_)
                        lambda1=lambda1_(a1);
                        lambda2=lambda2_(a2);
                        lambda3=lambda3_(a3);
                        lambda4=lambda4_(a4);
                        mu=mu_(a5);
                        anum=anum_(a);
                        clear D

                        %Normalize the training and testing images.
                        Xtrain=Xnormlize(double(Xtrain)); % training images
                        Y=Xnormlize(double(Y));% testing images
                        p_num=size(Xtrain,2)./length(Gtr); %patch size
                        
                        %Big memory needed.
                        fprintf('\nTraining......\n');
                        tic;
                        [D,U_concate,W,Acc_Tr,Acc_Tee,cost]=DADL_AOLP(Xtrain,L,anum,p_num,maxIter,lambda1,lambda2,lambda3,lambda4,Y,Gtr,Gte,mu);
                        trainingtime=toc;
                        fprintf('training time = %f\n',trainingtime);
                        
                        %number of testing images
                        Inum=size(Gte,2);
                        tic;
                        [Acc_Te,labely,Lte]=DADL_AOLP_Classifier(Y,D,W,Gte,Inum,p_num);
                        testingtime=toc;
                        fprintf('testing time = %f\n',testingtime);
                        fprintf('Classification Accuarcy = %f%% \n',Acc_Te*100);
                        % Drawing data
                        z(a1,a2)=Acc_Te;
                    end
                end
            end
        end
    end
end
%% 
bar3(z);
xlabel('\lambda_2');
ylabel('\lambda_1');
zlabel('classification accuracy');
b=[0 1];
set(gca,'zlim',b);
cc=['1e-4' ;'5e-4' ;'1e-3' ;'5e-3' ;'1e-2' ;'5e-2'];
dd=['1e-3' ;'5e-3' ;'1e-2' ;'5e-2' ;'1e-1' ;'5e-1'];
aa=['1e-5'; '1e-4' ;'1e-3' ;'1e-2' ;'1e-1' ;' 1  '];
bb=[' 1e-2' ;' 5e-2' ;' 1e-1' ;' 5e-1' ;'  1  ' ;'  5  '];

set(gca, 'XtickLabel', aa);
set(gca, 'YtickLabel', aa);
set(gca,'fontsize',15.5);


