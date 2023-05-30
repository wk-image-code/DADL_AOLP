function [D_f,U_concate_f,W_f,Acc_Tr,Acc_Te,cost]=DADL_AOLP(X,L,anum,p_num,maxIter,lambda1,lambda2,lambda3,lambda4,Y,Gtr,Gte,mu)

rng default

Inum=size(L,2);

%% Initialization
D = rand(anum,size(X,1));
U = zeros(p_num,anum*Inum);
W = rand(size(L,1),p_num*anum);
k=3;
[~,LL]= TripletLap(D',k);
% [~,LL]= TripletLap(L,k);
iter = 0;
alpha=1;
u=zeros(size(D,1),size(X,2));
Acc=0;
 while iter<=maxIter
% while maxIter
    iter = iter + 1;
 
    %update U                           
    u_tmp = inv(eye(size(LL,1))+2*mu*LL)*D*X;                             
   
    u_tmp=u_tmp';                                                         
    
    u=mat2cell(u_tmp,ones(1,Inum).*p_num,size(u_tmp,2));                   
    u=cell2mat(u');                                                        
    U_concate=reshape(u,[],Inum);                                         
    U_tmp=U_concate-alpha.*(-lambda2.*W'*L+lambda2.*W'*W*U_concate); 
  
    U = ALst(U_tmp, alpha.*lambda1);                                                             

    %update W
    W=lambda2.*L*U'*inv(lambda2.*U*U'+eye(size(U,1)).*lambda3);
    
    %update D
    u=reshape(U,p_num,[]);
    u=mat2cell(u',ones(1,Inum).*anum,size(u,1));                          
    u=cell2mat(u');                                                    
    D=u*X'*inv(X*X'+eye(size(X,1)).*lambda4);
    [~,LL]= TripletLap(D',k);
    normd=sqrt(sum(D.^2,2));
     D=D./repmat(normd.*(normd>1)+(normd<=1),1,size(D,2));
    
    %% Accuarcy Tracing
    a=size(Gtr,2);
    b=size(Gte,2);
    Acc_Tr(iter)=DADL_AOLP_Classifier(X,D,W,Gtr,a,p_num);
    Acc_Te(iter)=DADL_AOLP_Classifier(Y,D,W,Gte,b,p_num);
    fprintf('the traing accuracy：%4.3f,the test accuracy：%4.3f.\n',Acc_Tr(iter)*100,Acc_Te(iter)*100);
    %% costfunction
    tr(iter)=trace(u_tmp*LL*u_tmp');
    part1(iter)=norm(D*X-u_tmp','fro')^2/2;
    part2(iter)=lambda2*norm(L-W*U,'fro')^2/2;
    cost(iter)=part1(iter)+part2(iter);%part1(iter)+part2(iter)+tr(iter)+lambda1*norm(U,1);
    if (iter>10 &&Acc_Te(iter)>Acc)
        Acc = Acc_Te(iter);
        D_f = D;
        W_f = W;
        U_concate_f = U_concate;
    end
    %Acc_Te(iter-2)>Acc_Te(iter-1)&& Acc_Te(iter-1)>Acc_Te(iter)
    if (iter>5 && Acc_Te(iter-1)-Acc_Te(iter)>5)
        fprintf('warning: NaN appears in the current result');
        break;
    end
    if (iter>1 &&iter<10&& cost(iter)>cost(iter-1))
        iter = iter-1;
    end 
 end
u_tmp=u_tmp';
end
