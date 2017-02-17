function y=projsmplx(y)
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
end
