% Demo for DADL-AOLP
clear;close all;
load('./data/scene15.mat');
%% Hyper-parameters
lambda1_=[1e-4];
lambda2_=[1e-3];
lambda3_=[1e-5];
lambda4_=[1e-2];
mu_=[1];
anum_=[50];
maxIter=20;
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
                    end
                end
            end
        end
    end
end
%%
a=1:length(cost);
plot(a,cost(a),'*-','linewidth',2);
set(gca, 'YScale','log');
xlabel('Iteration number');
ylabel('Objective function value');



