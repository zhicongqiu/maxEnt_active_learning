function Fs_multiclass = getF_multiclassRare(data,GTT,label,Param,active_set)

M = size(data,1);

temp = find(active_set==1);
Fs_multiclass = zeros(M,length(active_set)+1);
for i=1:M
    sum = 1;
    for j=1:length(temp) %use the first k-1 classes
        sum = sum+exp(Param(temp(j)).beta0+data(i,:)*Param(temp(j)).beta');
    end
       
    for j=1:length(temp) %use the first k-1 classes
        Fs_multiclass(i,temp(j)) = ...
            exp(Param(temp(j)).beta0+data(i,:)*Param(temp(j)).beta')/sum;
        if Fs_multiclass(i,temp(j))>1-1e-10
            Fs_multiclass(i,temp(j))=1-1e-10;
        elseif Fs_multiclass(i,temp(j))<1e-10
            Fs_multiclass(i,temp(j))=1e-10;
        end
    end
    Fs_multiclass(i,end) = 1/sum;
    if Fs_multiclass(i,end)>1-1e-10
        Fs_multiclass(i,end)=1-1e-10;
    elseif Fs_multiclass(i,end)<1e-10
        Fs_multiclass(i,end)=1e-10;
    end        
end
            
        
    
