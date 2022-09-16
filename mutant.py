from snpmodel.dataset import amino_acid_alphabet, amino_acid_alphabet1to3
from snpmodel.dataset.preprocess import padding_fixLengthSeq, build_vocab_from_alphabet_dict
import re 
from Bio import SeqIO
from snpmodel.utils import *
from traitlets import Float 



def all_possible_mutant(protein_seq):
    out = []
    for origin, pos in zip(protein_seq, range(len(protein_seq))):
        pos_possible = []
        pos_3alphabet = amino_acid_alphabet1to3[origin]
        
        for pos_mutant, v in amino_acid_alphabet.items():
            pos_possible.append(f"{pos_3alphabet}{pos}{pos_mutant}")
        out.append(pos_possible)
    return out

def split_AAS(AAS):
    """
    AAS: like Val0Ala
    """
    
    AAS_list = re.split(r"([0-9]+)", AAS)
    origin = AAS_list[0] if AAS_list[0] else None  
    pos = int(AAS_list[1]) - 1 if AAS_list[1] else None
    mutant = AAS_list[2] if AAS_list[2] else None
    
    return origin, pos, mutant



def all_possible_mutant_sequence(protein_seq, fix_length = None, padding="<pad>"):
    """
    输入蛋白序列，获得所有突变的序列情况（AAS）， fix_length和padding必须一起使用，fix_length 应当输入2的倍数
    
    Return 
        list， 每一个突变序列均是list保存，返回是所有突变序列的list对象的集合
    """
    all_possible_mutant_hgvs = all_possible_mutant(protein_seq)
    out = []
    hgvs_label = []
    for pos_hgvs in all_possible_mutant_hgvs:  # 按每个位置进行迭代，
        for hgvs in pos_hgvs:
            origin, pos, mutant = split_AAS(hgvs)
            mutant_seq = list(changeStrByPos(protein_seq, pos, amino_acid_alphabet[mutant]))  # 转化成list类型

            if fix_length and padding:
                mutant_seq = padding_fixLengthSeq(mutant_seq, pos, fix_length, padding)  # 按指定长度进行截取

            out.append(mutant_seq)
            hgvs_label.append(hgvs)
        
    return out, hgvs_label


# get context array


dbNSFP = ["GERP++_NR", "GERP++_RS", "phyloP100way_vertebrate", "phyloP30way_mammalian", "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons30way_mammalian", "phastCons17way_primate", "1000Gp3_AF", "UK10K_AF"]
clinvar = ["clinvar_id", "clinvar_hgvs", "clinvar_review", "clinvar_clnsig"]
basic = ["rs_dbSNP", "aaref", "aaalt", "aapos", "Uniprot_acc", "HGVSp_ANNOVAR"]


# 加载uniprot acc在uniprot 上下载得到的序列
def parase_Uniprot(fasta):
    datas = SeqIO.parse(fasta,"fasta")
    return pd.DataFrame([{"uniprot accession":i.id.split("|")[1], "uniprot sequence": str(i.seq)} for i in datas])


def ref_seq_context_df(seq:str, alphabet = amino_acid_alphabet1to3, as_index=True):
    """
    ref_seq_context_df 输入蛋白质序列，返回一个df，三列

    Args:
        seq (str): _description_
        alphabet (_type_, optional): _description_. Defaults to amino_acid_alphabet1to3。字典类型{"A":"Arg", ...}，生成的每个pos的顺序按照key的顺序进行排序

    Returns:
        _type_: df，三列：aapos, aaalt, aaref； as_index =True，则把aapos和aaalt作为索引；aapos是1-base；aaalt的顺序按照alphabet keys的顺序。
    """
    if as_index:
        return pd.DataFrame([{"aapos": pos, "aaalt": alt, "aaref":ref} for pos, ref in zip(range(1, len(seq)+1),list(seq)) for alt in alphabet.keys()]).set_index(["aapos", "aaalt"])
    else:
        return pd.DataFrame([{"aapos": pos, "aaalt": alt, "aaref":ref} for pos, ref in zip(range(1, len(seq)+1),list(seq)) for alt in alphabet.keys()])
def loadAssociateGeneSNP_ByUniprotAcc(uniprotAcc:str, uniprot:pd.DataFrame, context_path:str = "data/filter_data/associateGeneSNP_dbNSFP", others=False):
    """
    loadAssociateGeneSNP_ByUniprotAcc 根据acc，和uniprot的蛋白质序列

    Args:
        uniprotAcc (str): uniprot accession
        uniprot (pd.DataFrame): 两列：uniprot accession和uniprot sequence
        context_path (str): dbNSFP检索出来的结果，按uniprot accession命名的csv文件目录
    Returns:
        _type_: 传出包含：dbNSFP = ["GERP++_NR", "GERP++_RS", "phyloP100way_vertebrate", "phyloP30way_mammalian", "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons30way_mammalian", "phastCons17way_primate", "1000Gp3_AF", "UK10K_AF"];clinvar = ["clinvar_id", "clinvar_hgvs", "clinvar_review", "clinvar_clnsig"];basic = ["rs_dbSNP", "aaref", "aaalt", "aapos", "Uniprot_acc", "HGVSp_ANNOVAR"]的dataframe。 

        排除了含有”X“未知氨基酸以及结尾终止突变产生新的氨基酸的突变以及可变剪切突变数据
    Exapmle:
        ```python
        context_snps, seq = loadAssociateGeneSNP_ByUniprotAcc(uniprotAcc, uniprot)
        ```
    """

    # context_path = "data/filter_data/associateGeneSNP_dbNSFP"
    context_SNPs_path = os.path.join(context_path, uniprotAcc + ".csv") 

    dbNSFP = ["GERP++_NR", "GERP++_RS", "phyloP100way_vertebrate", "phyloP30way_mammalian", "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons30way_mammalian", "phastCons17way_primate", "1000Gp3_AF", "UK10K_AF"]
    clinvar = ["clinvar_id", "clinvar_hgvs", "clinvar_review", "clinvar_clnsig"]
    basic = ["rs_dbSNP", "aaref", "aaalt", "aapos", "Uniprot_acc", "HGVSp_ANNOVAR"]

    print(f"loading data from {context_SNPs_path}")
    # try:
    context_snps = pd.read_csv(context_SNPs_path, sep = "\t")
    # except:
    #     raise FileNotFoundError
    if not others:
        context_snps = context_snps[basic + dbNSFP + clinvar]
    if others:
        context_snps = context_snps[basic + dbNSFP + clinvar + others]



    seq = uniprot[uniprot["uniprot accession"]==uniprotAcc]["uniprot sequence"].values[0]
    seq_length = len(seq)

    context_snps = context_snps[context_snps["aaref"] != "X" ]  #排除含有未知氨基酸的数据
    context_snps = context_snps[context_snps["aaalt"] != "X" ]  #排除含有未知氨基酸的数据
    # 替换所有“.”和nan为0

    context_snps.loc[:, dbNSFP] = context_snps.loc[:, dbNSFP].replace(".", 0)
    # context_snps.loc[:, dbNSFP] = context_snps.loc[:, dbNSFP].fillna(0).astype(float) 

    def func(x:pd.Series, uniprotAcc):
        listPos = searchList(x["Uniprot_acc"].split(";"), uniprotAcc)
        x["aapos"] = str(x["aapos"]).split(";")[listPos] if listPos is not None else str(x["aapos"]).split(";")[0]
        x["Uniprot_acc"] = x["Uniprot_acc"].split(";")[listPos] if listPos is not None else x["Uniprot_acc"].split(";")[0]
        x["HGVSp_ANNOVAR"] = x["HGVSp_ANNOVAR"].split(";")[listPos] if listPos is not None else x["HGVSp_ANNOVAR"].split(";")[0]
        return x
    # 对应位置
    try:
        context_snps = context_snps.apply(lambda x: func(x, uniprotAcc),axis = 1)
    except:
        print(f"int error {uniprotAcc}")
    context_snps["aapos"] = context_snps["aapos"].astype(int)  #更改数据类型为int

    context_snps = context_snps[context_snps["aapos"].apply(lambda x: x <= seq_length and x >=1) ]  # 去掉长度大于参考序列的变异，这部分通常是终止突变成某个氨基酸; -1 为可变剪切发生，去掉
    #  对应acc的序列


    ref_df = ref_seq_context_df(seq)

    return context_snps, seq

def generate_context_df(context_snps:pd.DataFrame, seq:str, as_index=False, fillna=0, alphabet=amino_acid_alphabet1to3):
    """
    generate_context_df 生成包含已知SNP和未知SNP的所有可能突变df，

    Args:
        context_snps (pd.DataFrame): loadAssociateGeneSNP_ByUniprotAcc 返回的结果
        seq (str): uniprot accession对应的uniprot sequence
        as_index (bool, optional): 是否返回index. Defaults to False.
        fillna (int, optional): _description_. Defaults to 0.
        alphabet (_type_, optional): _description_. Defaults to amino_acid_alphabet1to3。字典类型{"A":"Arg", ...}

    Returns:
        _type_: 返回包含已知和未知SNP的df
    Exapmle:
        ```python
        context_full_df = generate_context_df(context_snps, seq, alphabet=alphabet, fillna=fillna)        
        ```

    """
    context_snps = context_snps.set_index(["aapos", "aaalt"])
    merge = context_snps.merge(ref_seq_context_df(seq, alphabet=alphabet), how = "outer", left_index=True, right_index=True, )
    merge = merge[~merge.index.duplicated(keep="first")]
    merge.insert(1, "aaref", merge["aaref_y"])
    merge.drop(columns=["aaref_x", "aaref_y"], inplace=True)

    merge.loc[:, dbNSFP] = merge.loc[:,dbNSFP].fillna(0).astype(float) 

    if as_index:
        pass 
    elif not as_index:
        merge.reset_index(drop=False, inplace=True )

    return merge 


#  get context data 

def get_uniprot_Acc_numpyArray(uniprotAcc, uniprot, alphabet, fillna=0):
    """
    get_uniprot_Acc_numpyArray 传入uniprotAcc和uniprotacc~seq的df以及alphabet，生成对应的uniprotAcc氨基酸序列空间内所有变异以及对应的各类分数的矩阵。
    第一列是aapos是1-base；第二列是aaalt的顺序按照alphabet keys的顺序。

    Args:
        uniprotAcc (_type_): _description_
        uniprot (_type_): _description_
        alphabet (_type_): _description_

    Returns:
        _type_: _description_
    Exapmles:
        ``` python
        alphabet = amino_acid_alphabet1to3
        #  读取数据
        clinvar_data = pd.read_csv("data/filter_data/clinvar_uniprot_seq.csv", sep="\t")

        fasta = "data/filter_data/uniprot.fasta"
        uniprot = parase_Uniprot(fasta)

        #  获取感兴趣的蛋白
        uniprotAcc = clinvar_data["uniprot accession"].unique()[1]
        print(uniprotAcc)
        clinvar_data[clinvar_data["uniprot accession"] == uniprotAcc]
        
        get_uniprot_Acc_numpyArray(uniprotAcc, uniprot, alphabet=alphabet)
        ```

    """
    context_snps, seq = loadAssociateGeneSNP_ByUniprotAcc(uniprotAcc, uniprot)
    context_full_df = generate_context_df(context_snps, seq, alphabet=alphabet, fillna=fillna)


    context_array = context_full_df.loc[:, ["aapos", "aaalt"] + dbNSFP].to_numpy().reshape(len(seq), len(alphabet.keys()), -1)  # 生成context_array
    return context_array
