function Dbeta = get_Dbeta(data,Fa,Fs_multiclass,data_GTT_temp,label_temp,l,active_set,num,numN,numR,a_u,rare)

sum0 = 0;
sum1 = 0;
sum2 = 0;
for i=1:size(Fs_multiclass,1)
    if label_temp(i)~=2 %labeled class
        if data_GTT_temp(i,1)==l %same-class label
            sum0 = sum0-(1-Fs_multiclass(i,l))*data(i);
        else %diff-class label
            sum1 = sum1+Fs_multiclass(i,l)*data(i);
        end
    else %unknown class
        temp = find(active_set==1);
        for j=1:length(temp)
            if temp(j)==l
                sum2 = sum2-Fa(i)*(1-Fs_multiclass(i,l))*data(i);
            else
                sum2 = sum2+Fa(i)*Fs_multiclass(i,l)*data(i);
            end
        end
        %add-up the unknown
        if rare==true
            sum2 = sum2+Fa(i)*Fs_multiclass(i,end)*data(i);
        end
    end
end

if rare==true
    Dbeta = (1-a_u)*(numN/numR)*(sum0+sum1)+a_u*sum2/(1+num);
else
    Dbeta = (1-a_u)*(sum0+sum1)+a_u*sum2/num;
end

end