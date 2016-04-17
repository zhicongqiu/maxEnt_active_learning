function [PMFunknown PMFnormal PMFrare] = getTestPMF(test_raw,test_p,test_GTT,ParamPR,ParamN,ParamR,active_set_normal,active_set_rare,pval)
%evaluate posterior probabilities

M = size(test_raw,1);

PMFunknown = getF(test_p,ParamPR);

if sum(active_set_normal)>=2
PMFnormal = getF_multiclassNormal(test_raw,test_GTT,2*ones(M,1),ParamN,active_set_normal);
else
    PMFnormal = zeros(M,length(active_set_normal));
    PMFnormal(:,active_set_normal==1) = 1;
end
if sum(active_set_rare)>=1
    if pval==false
        PMFrare = getF_multiclassRare(test_raw,test_GTT,2*ones(M,1),ParamR,active_set_rare);
    else
        PMFrare = getF_multiclassRare(test_p,test_GTT,2*ones(M,1),ParamR,active_set_rare);
    end
else
    PMFrare = zeros(M,length(active_set_rare)+1);
end
end
    