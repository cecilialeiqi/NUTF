clear
clc

days=20;
time_step=10;
T=time_step*days
C=200;
cat_person=10; % 10 kinds of different people
N=1000;
%sparsity=10; % 10 percent of check in data
% # of rows: m=100, cat_person(5)*people_per_cat(20)
if (mod(N,cat_person)~=0)
	fprintf('Error: user number must be a multiple of people categories\n')
	return;
end

% generate random living patterns
rng(0,'twister');
arr=randi([1,C],cat_person,time_step);
M=T*C; % column
people_per_cat=N/cat_person;

p=4; % # of possible locations

nnz_matrix=0; %groundTruth

nnz_train=0;

% define Omega:
Omega=[];

i_matrix=repmat(1:N,[T,1]);
i_matrix=reshape(i_matrix,[N*T,1]);

fprintf('generating data!\n');
j_matrix=repmat(arr,[1,days]);
j_matrix=j_matrix+repmat((0:(T-1))*C,[cat_person,1]);
j_matrix=repmat(j_matrix',[1,people_per_cat]);
j_matrix=reshape(j_matrix,[N*T,1]);

i_train=repmat(1:N,[p*T,1]);
i_train=reshape(i_train,[p*N*T,1]);
j_train_noise=randi([1,C],N*T*p,1);
a=repmat(reshape(repmat((0:(T-1))*C,[p,1]),T*p,1),[N,1]);
j_train=j_train_noise+a;
j_train(1:p:N*T*p)=j_matrix;
Omega=reshape(1:N*T*p,[p,N*T]);
Omega=Omega';


% example: [  4 (5) 6 | 1 2 (3) | 1 (2) 9 10 |  (1) 8 9 | 2 (7) 10| 4 (5) 8 | repeat for 7 days.. ]   
% Omega{1-42}: first week
matrix=sparse(i_matrix,j_matrix,ones(N*T,1),N,M);
train=sparse(i_train,j_train,ones(N*T*p,1)*1.0/p,N,M);
fprintf('Finish generating data!\n');

r=10;%min(m,n);
%r=15;
iter=10;
X=train;
%X=sparse(X);
u=zeros(N,1); v=zeros(M,1);
[U,S,V]=randomsvd(X,u,v,N,M,r,[],6);
%Y=U*S*V';
val=dot(U(i_train,:)*S,V(j_train,:),2);

for it=1:iter
	
	% project to simplex:
	tic
	for j=1:N
		for k=1:T
			ind=Omega((j-1)*T+k,:);
			tmp=val(ind);%Y(j,ind);
			val(ind)=projsmplx(tmp);
		end
	end
	% project to zeros:
	%X=Y.*train;
	Y=sparse(i_train,j_train,val,N,M);

	time1=toc;
	
	tic
	[U,S,V]=randomsvd(Y,u,v,N,M,r,[],6);
	
	time2=toc;
	%Y=U*S*V';
	val2=dot(U(i_train,:)*S,V(j_train,:),2);
	f=norm(val2-val);%norm(X-Y,'fro');
	val=val2;
	fprintf('projection:%.3f sec, rsvd:%.3f sec, f(X,Y):%.4f\n',time1,time2,f);
	if (it>1 && f>lastf-0.01)
		break;
	end
	lastf=f;
	
end

pred=Y;
IP=sum(sum(pred.*matrix))/sqrt(sum(sum(pred))*sum(sum(matrix)));
fprintf('Inner product with ground truth:%.3f\n', nonzeros(IP));
for i=1:N*T
	ind=Omega(i,:);
	if (isempty(ind))
		continue;
	end
	j=floor((i-1)/T)+1;
	a=val(ind);
	[b,c]=max(a);
	i_pred(i)=j;
	j_pred(i)=j_train(ind(c));%ind(c);
end
pred_thres=sparse(i_pred,j_pred,ones(length(i_pred),1),N,M);
acc=sum(sum(pred_thres.*matrix))/sqrt(sum(sum(pred_thres))*sum(sum(matrix)));
fprintf('Accuracy=%.3f',nonzeros(acc))

