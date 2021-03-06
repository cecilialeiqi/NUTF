#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include "city.h"

#pragma GCC diagnostic ignored "-Wwrite-strings"
using namespace std;
typedef Eigen::SparseMatrix<double,RowMajor> SpMat;
typedef Eigen::SparseMatrix<double,ColMajor> SpMatc;
typedef Eigen::Triplet<int> Tp;

void read(vector<Tp>& A_in_triplets, char* filename)
{
	ifstream f(filename);
	long num=0;
	long i,j;
	clock_t st = clock();
    long maxn=0;
	long maxm=0;
	while (!f.eof()){
		f>>i;
		if (f.eof()) break;
		f>>j;
		if (f.eof()) break;
		num++;
		A_in_triplets.push_back(Tp(i,j,1));
		if (num-long(num/1000000)*1000000==0)
		{
			cout<<"Reading "<<num<<": "<<i<<' '<<j<<" 1"<<endl;
		}
		if (i>maxn)
			maxn=i;
		if (j>maxm)
			maxm=j;
	}
	cout<<"maxn="<<maxn<<endl;
	cout<<"maxm="<<maxm<<endl;
	clock_t end = clock();
	cout<<"reading file time: "<<double(end-st)/CLOCKS_PER_SEC<<endl;
}

void construct_support(vector<Tp> & A_in_triplets, vector<set<int> > & Omega, int N, int T, int C, SpMat & X){
	for (int i=0;i<A_in_triplets.size(); i++){
		int x=A_in_triplets[i].row();
		int y=A_in_triplets[i].col();
		Omega[x*T+y/C].insert(y);
	}
	//long nnz=A_in_triplets.size();

	/*
	 * loop through time t \in [T],
	 * then loop through users \in [N], do
	 * 1) record the users that doesn't have any info at this time
	 * 2) meanwhile, make a set that includes all the locations that are visited in this time
	 * Afterwards, go to the missing users, and link the user to the visited place */
	/*for (int t=0;t<T; t++)
	{
		vector<int> empty_users;
		//set<int> visited_cat;
		for (int i=0;i<N;i++){
			int ind=i*T+t;
			if (Omega[ind].empty())
				empty_users.push_back(i);
			else{
				for (auto j:Omega[ind]){
					visited_cat.insert(j%C);
					//X.insert(i,j)=1.0/Omega[ind].size();
					//nnz++;
				}
			}
		}
		//int s=visited_cat.size();
		for (int i=0;i<empty_users.size();i++){
			int ind=empty_users[i]*T+t;
			for (auto c:visited_cat){
				Omega[ind].insert(c+t*C);
				A_in_triplets.push_back(Tp(empty_users[i],t*C+c,1));
				//X.insert(empty_users[i],t*C+c)=1.0/s;
				if (t*C+c>T*C)
				{
					cout<<"t="<<t<<endl;
					cout<<"c="<<c<<endl;
					cout<<"n="<<empty_users[i];
					exit(0);
				}

			}
		}
	}*/

	cout<<"nnz of input matrix is: "<<A_in_triplets.size()<<endl;
	X.setFromTriplets(A_in_triplets.begin(), A_in_triplets.end());
}

void GS_QR(MatrixXd &Q, const MatrixXd & Y){
	double a;
	for (int j=0;j<Y.cols();j++){
		for (int i=0;i<j;i++){
			a=Q.col(i).dot(Y.col(j));
			Q.col(j)-=a*Q.col(i);
		}
		a=Q.col(j).norm();
		Q.col(j)/=a;
	}
}

void randomSVD(int N, int M, SpMat &X, int K, int maxit)
{//K is rank
/*function [U S V] = randomsvd(A,uu,vv, m,n,k, INIT, maxit)
 *
 * istrans = 0;
 * if numel(INIT) == 0
 *     Omega = randn(n, k);
 * else
 *     tt = min(k,size(INIT,2));
 *     Omega = INIT(:,1:tt);
 *     if tt < k
 *         Omega = [Omega randn(n,k-tt)];
 *     end
 * end
 * Y = A*Omega+uu*(vv'*Omega);
 * Atrans = A';
 * [Q,~] = comp_qr(Y);
 * for i=1:maxit
 *     BB = Atrans*Q + vv*(uu'*Q);
 *     Y = A*BB + uu*(vv'*BB);
 *     [Q, ~] = comp_qr(Y);
 * end
 * [Q r] = qr(Y,0);
 * B = Q'*A+(Q'*uu)*vv';
 * [u S V] = svd(B,'econ');
 * U = Q*u;*/
	bool transpose;
	//double st=clock();
	SpMatc A, Atrans;
	if (N>M)
	{
		Atrans=X;
		A=X.transpose();
		transpose=true;
	}else{
		Atrans=X.transpose();
		A=X;
		transpose=false;
	}
	//cout<<"transform costs: "<<clock()-st<<" seconds"<<endl;
	//st=clock();
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0,1);
	auto gaussian=[&](double) {return distribution(generator);};
	MatrixXd Omega=MatrixXd::NullaryExpr(A.cols(),K,gaussian);
	//cout<<"generate random matrix costs: "<<clock()-st<<" seconds"<<endl;
	//st=clock();
	MatrixXd Y=A*Omega;
	//cout<<"Y=A*Omega costs: "<<clock()-st<<" seconds"<<endl;
	//st=clock();
	/*HouseholderQR<MatrixXd> qr(Y);
	MatrixXd QQ=qr.householderQ();
	MatrixXd Q=QQ.block(0,0,Y.rows(),K);*/
	MatrixXd Q=Y;
	GS_QR(Q,Y);
	//cout<<Q.cols()<<','<<Q.rows()<<endl;
	MatrixXd BB;
	//cout<<"QR factorization costs: "<<clock()-st<<" seconds"<<endl;
	//st=clock();
	for (int i=0;i<maxit;i++){
		BB=Atrans*Q;
		//cout<<"BB=Atrans*Q cost: "<<clock()-st<<" seconds"<<endl;
		//st=clock();
		Y=A*BB;
		//cout<<"Y=A*BB cost: "<<clock()-st<<" seconds"<<endl;
		//st=clock();
		/*HouseholderQR<MatrixXd> qr(Y);
		QQ=qr.householderQ();
		Q=QQ.block(0,0,Y.rows(),K);*/
		Q=Y;
		GS_QR(Q,Y);
		//cout<<"two QR factorizations cost: "<<clock()-st<<" seconds"<<endl;
		//st=clock();
	}
	BB=Atrans*Q;
	if (transpose)
	{
		// A ~ BB'*Q'
		//A[i,j]=BB[:,i]*Q[j,:]
		for (int i=0;i<N;i++)
			for (SparseMatrix<double, RowMajor>::InnerIterator it(X,i); it; ++it)
				X.coeffRef(i,it.col())=BB.row(i).dot(Q.row(it.col()));
	}else{//A ~ Q*BB
		for (int i=0;i<N;i++)
			for (SparseMatrix<double, RowMajor>::InnerIterator it(X,i); it; ++it)
				X.coeffRef(i,it.col())=Q.row(i).dot(BB.row(it.col()));
	}
}
/*
void output(SpMat &X, vector<set<int>> & Omega, int N, int T, int C, vector<vector<int>> & result){
	for (int n=0;n<N;n++){
		for (int t=0;t<T;t++){
			int ind=n*T+t;
			//Omega[ind] indicates the coordinates of possible categories
			result[n][t]=0;
			double m=0;
			for (auto it:Omega[ind]){
				if (X[n][it]>m){
					m=X[n][it];
					result[n][t]=it%C;
				}
			}
		}

	}
}*/


void smplxproj(double & diff, SpMat & X, set<int> & omega, int n){
	/*
	 b=sort(y);
	 n=length(y);
	 t=zeros(n,1);
	 tmp=b(n);
	 for i=n-1:-1:0
	 	if i==0
			that=(sum(b)-1)/n;
			break;
		end
		t(i)=(tmp-1)/(n-i);
		tmp=tmp+b(i);
		if (t(i)>=b(i))
			that=t(i);
			break;
		end
	end
	y=max(y-that,0);
	 */
	vector<double> y;
	for (auto i:omega){
		y.push_back(X.coeffRef(n,i));
	}
	int l=omega.size();
	vector<double> b=y;
	std::sort(b.begin(),b.end());
	vector<double> t(l,0);
	double tmp=b[l-1];
	double that=0;
	for (int i=l-1; i>=0; i--){
		if (i==0){
			for (int j=0;j<l;j++)
				that+=b[j];
			that=(that-1)/l;
			break;
		}
		t[i-1]=(tmp-1)/(l-i);
		tmp+=b[i-1];
		if (t[i-1]>=b[i-1]){
			that=t[i-1];
			break;
		}
	}
	set<int>::iterator it=omega.begin();
	for (int i=0;i<l;i++){
		double newvalue=max(y[i]-that,0.0);
		//cout<<newvalue<<",("<<n<<","<<*it<<"),";
		diff+=(y[i]-newvalue)*(y[i]-newvalue);
		X.coeffRef(n,*it)=newvalue;
		++it;
	}
	//cout<<endl;
}


void model(SpMat & X, int iter, int N, int T, int C, int K, vector<set<int>> & Omega)
{
	/* Matlab code
	tic
	for j=1:m
		for k=1:n/cat_place
			ind=Omega{(j-1)*n/cat_place+k};
			tmp=Y(j,ind);
			Y(j,ind)=projsmplx(tmp);
		end
	end
	% project to zeros:
	X=Y.*train;
	time1=toc;
	tic
	[U,S,V]=randomsvd(X,u,v,m,n,r,[],6);
	time2=toc;
	Y=U*S*V';
	f=norm(X-Y,'fro');
	fprintf('projection:%.3f, rsvd:%.3f, f(X,Y):%.4f\n',time1,time2,f);
	if (it>1 && f>lastf-0.01)
		break;
	end
	lastf=f;*/

	// first need to set up X with support in Omega
	//
	for (int i=0; i<iter;i++){
		// first, calculate randomSVD of X and projected to
		double diff=0;
		//randomSVD(int N, int M, SpMat &X, int K, int maxit)
		double st=clock();
		randomSVD(N, T*C, X, K, 6);
		cout<<"Finished random SVD, used "<<(clock()-st)/CLOCKS_PER_SEC<<" seconds"<<endl;
		//Y=X;
		st=clock();
		for (int i=0;i<Omega.size();i++){
			if (Omega[i].empty())
				continue;
			int n=i/T;
			int t=i%T;
			if (Omega[i].size()==1)
			{
				double oldvalue=X.coeffRef(n, *Omega[i].begin());
				X.coeffRef(n,*Omega[i].begin())=1;
				diff+=(1-oldvalue)*(1-oldvalue);
			}else{
				smplxproj(diff, X, Omega[i], n);
			}
		}
		cout<<"Finished simplex projection, used "<<(clock()-st)/CLOCKS_PER_SEC<<" seconds"<<endl;
		cout<<"Iteration "<<i<<": error="<<sqrt(diff)<<endl;
	}
}

int main(int argc, char* argv[]){
	if (argc!=7){
		cout<<"usage: "<<argv[0]<<" <filename> <N> <T> <C> <r> <iter>"<<endl;
		exit(0);
	}
	char * filename=argv[1];
	int N=atoi(argv[2]);
	int T=atoi(argv[3]);
	int C=atoi(argv[4]);
	int r=atoi(argv[5]);
	int iter=atoi(argv[6]);
	vector<int> groundTruth(N*T,0);
	ifstream f(filename);
	SpMat A(N,T*C);
	cout<<A.cols()<<endl;
	cout<<A.rows()<<endl;
	vector<Tp> A_in_triplets;
	read(A_in_triplets, filename);
	A.setFromTriplets(A_in_triplets.begin(), A_in_triplets.end());
	cout<<"Finish initialization with the observed matrix!"<<endl;
	vector<set<int>> Omega(N*T,set<int>());
	SpMat X(N,T*C);
	construct_support(A_in_triplets,Omega,N, T, C, X);
	cout<<"Finish initialization with the input matrix!"<<endl;
	model(X,iter,N,T,C,r,Omega);
	ofstream of("output.csv");
	for (int i=0;i<N;i++)
		for (SparseMatrix<double, RowMajor>::InnerIterator it(X,i); it; ++it)
			of<<i<<','<<it.col()<<','<<it.value()<<endl;
	return 0;
}

