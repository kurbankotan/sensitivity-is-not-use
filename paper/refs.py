"""Verified reference entries. Every one was checked against OpenAlex or arXiv in this session;
any that could not be verified is absent rather than guessed."""
REFS = {
 "vina": "Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed and accuracy of docking with a "
         "new scoring function, efficient optimization, and multithreading. Journal of Computational "
         "Chemistry, 31(2), 455-461. https://doi.org/10.1002/jcc.21334",
 "rfscore": "Ballester, P. J., & Mitchell, J. B. O. (2010). A machine learning approach to predicting "
            "protein-ligand binding affinity with applications to molecular docking. Bioinformatics, 26(9), "
            "1169-1175. https://doi.org/10.1093/bioinformatics/btq112",
 "ragoza": "Ragoza, M., Hochuli, J., Idrobo, E., Sunseri, J., & Koes, D. R. (2017). Protein-ligand scoring "
           "with convolutional neural networks. Journal of Chemical Information and Modeling, 57(4), 942-957. "
           "https://doi.org/10.1021/acs.jcim.6b00740",
 "pafnucy": "Stepniewska-Dziubinska, M. M., Zielenkiewicz, P., & Siedlecki, P. (2018). Development and "
            "evaluation of a deep learning model for protein-ligand binding affinity prediction. "
            "Bioinformatics, 34(21), 3666-3674. https://doi.org/10.1093/bioinformatics/bty374",
 "kdeep": "Jimenez, J., Skalic, M., Martinez-Rosell, G., & De Fabritiis, G. (2018). KDEEP: Protein-ligand "
          "absolute binding affinity prediction via 3D-convolutional neural networks. Journal of Chemical "
          "Information and Modeling, 58(2), 287-296. https://doi.org/10.1021/acs.jcim.7b00650",
 "onionnet": "Zheng, L., Fan, J., & Mu, Y. (2019). OnionNet: A multiple-layer intermolecular-contact-based "
             "convolutional neural network for protein-ligand binding affinity prediction. ACS Omega, 4(14), "
             "15956-15965. https://doi.org/10.1021/acsomega.9b01997",
 "crossdocked": "Francoeur, P. G., Masuda, T., Sunseri, J., Jia, A., Iovanisci, R. B., Snyder, I., & Koes, "
                "D. R. (2020). Three-dimensional convolutional neural networks and a cross-docked data set "
                "for structure-based drug design. Journal of Chemical Information and Modeling, 60(9), "
                "4200-4215. https://doi.org/10.1021/acs.jcim.0c00411",
 "mpnn": "Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message "
         "passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine "
         "Learning. arXiv:1704.01212. https://arxiv.org/abs/1704.01212",
 "schnet": "Schutt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A., & Muller, K.-R. (2018). SchNet: "
           "A deep learning architecture for molecules and materials. The Journal of Chemical Physics, "
           "148(24), 241722. https://doi.org/10.1063/1.5019779",
 "egnn": "Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) equivariant graph neural networks. In "
         "Proceedings of the 38th International Conference on Machine Learning. arXiv:2102.09844. "
         "https://arxiv.org/abs/2102.09844",
 "sign": "Li, S., Zhou, J., Xu, T., Huang, L., Wang, F., Xiong, H., Huang, W., Dou, D., & Xiong, H. (2021). "
         "Structure-aware interactive graph neural networks for the prediction of protein-ligand binding "
         "affinity. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining "
         "(pp. 975-985). https://doi.org/10.1145/3447548.3467311",
 "gign": "Yang, Z., Zhong, W., Lv, Q., Dong, T., & Chen, C. Y.-C. (2023). Geometric interaction graph neural "
         "network for predicting protein-ligand binding affinities from 3D structures (GIGN). The Journal of "
         "Physical Chemistry Letters, 14(8), 2020-2033. https://doi.org/10.1021/acs.jpclett.2c03906",
 "pignet": "Moon, S., Zhung, W., Yang, S., Lim, J., & Kim, W. Y. (2022). PIGNet: A physics-informed deep "
           "learning model toward generalized drug-target interaction predictions. Chemical Science, 13(13), "
           "3661-3673. https://doi.org/10.1039/D1SC06946B",
 "hacnet": "Kyro, G. W., Brent, R. I., & Batista, V. S. (2023). HAC-Net: A hybrid attention-based "
           "convolutional neural network for highly accurate protein-ligand binding affinity prediction. "
           "Journal of Chemical Information and Modeling, 63(7), 1947-1960. "
           "https://doi.org/10.1021/acs.jcim.3c00251",
 "rtmscore": "Shen, C., Zhang, X., Deng, Y., Gao, J., Wang, D., Xu, L., Pan, P., Hou, T., & Kang, Y. (2022). "
             "Boosting protein-ligand binding pose prediction and virtual screening based on residue-atom "
             "distance likelihood potential and graph transformer. Journal of Medicinal Chemistry, 65(15), "
             "10691-10706. https://doi.org/10.1021/acs.jmedchem.2c00991",
 "deepdta": "Ozturk, H., Ozgur, A., & Ozkirimli, E. (2018). DeepDTA: Deep drug-target binding affinity "
            "prediction. Bioinformatics, 34(17), i821-i829. https://doi.org/10.1093/bioinformatics/bty593",
 "graphdta": "Nguyen, T., Le, H., Quinn, T. P., Nguyen, T., Le, T. D., & Venkatesh, S. (2021). GraphDTA: "
             "Predicting drug-target binding affinity with graph neural networks. Bioinformatics, 37(8), "
             "1140-1147. https://doi.org/10.1093/bioinformatics/btaa921",
 "equibind": "Stark, H., Ganea, O.-E., Pattanaik, L., Barzilay, R., & Jaakkola, T. (2022). EquiBind: "
             "Geometric deep learning for drug binding structure prediction. In Proceedings of the 39th "
             "International Conference on Machine Learning. arXiv:2202.05146. "
             "https://arxiv.org/abs/2202.05146",
 "diffdock": "Corso, G., Stark, H., Jing, B., Barzilay, R., & Jaakkola, T. (2023). DiffDock: Diffusion steps, "
             "twists, and turns for molecular docking. In International Conference on Learning "
             "Representations. arXiv:2210.01776. https://arxiv.org/abs/2210.01776",
 "masters": "Masters, M. R., Mahmoud, A. H., Wei, Y., & Lill, M. A. (2023). Deep learning model for efficient "
            "protein-ligand docking with implicit side-chain flexibility. Journal of Chemical Information and "
            "Modeling, 63(6), 1695-1707. https://doi.org/10.1021/acs.jcim.2c01436",
 "casf": "Su, M., Yang, Q., Du, Y., Feng, G., Liu, Z., Li, Y., & Wang, R. (2019). Comparative assessment of "
         "scoring functions: The CASF-2016 update. Journal of Chemical Information and Modeling, 59(2), "
         "895-913. https://doi.org/10.1021/acs.jcim.8b00545",
 "gabel": "Gabel, J., Desaphy, J., & Rognan, D. (2014). Beware of machine learning-based scoring functions: "
          "On the danger of developing black boxes. Journal of Chemical Information and Modeling, 54(10), "
          "2807-2815. https://doi.org/10.1021/ci500406k",
 "liyang": "Li, Y., & Yang, J. (2017). Structural and sequence similarity makes a significant impact on "
           "machine-learning-based scoring functions for protein-ligand interactions. Journal of Chemical "
           "Information and Modeling, 57(4), 1007-1012. https://doi.org/10.1021/acs.jcim.7b00049",
 "wallach": "Wallach, I., & Heifets, A. (2018). Most ligand-based classification benchmarks reward "
            "memorization rather than generalization. Journal of Chemical Information and Modeling, 58(5), "
            "916-932. https://doi.org/10.1021/acs.jcim.7b00403",
 "dude": "Mysinger, M. M., Carchia, M., Irwin, J. J., & Shoichet, B. K. (2012). Directory of Useful Decoys, "
         "Enhanced (DUD-E): Better ligands and decoys for better benchmarking. Journal of Medicinal "
         "Chemistry, 55(14), 6582-6594. https://doi.org/10.1021/jm300687e",
 "dudebias": "Chen, L., Cruz, A., Ramsey, S., Dickson, C. J., Duca, J. S., Hornak, V., Koes, D. R., & "
             "Kurtzman, T. (2019). Hidden bias in the DUD-E dataset leads to misleading performance of deep "
             "learning in structure-based virtual screening. PLOS ONE, 14(8), e0220113. "
             "https://doi.org/10.1371/journal.pone.0220113",
 "scantlebury": "Scantlebury, J., Brown, N., von Delft, F., & Deane, C. M. (2020). Data set augmentation "
                "allows deep learning-based virtual screening to better generalize to unseen target classes "
                "and highlight important binding interactions. Journal of Chemical Information and Modeling, "
                "60(8), 3722-3730. https://doi.org/10.1021/acs.jcim.0c00263",
 "sieg": "Sieg, J., Flachsenberg, F., & Rarey, M. (2019). In need of bias control: Evaluating chemical data "
         "for machine learning in structure-based virtual screening. Journal of Chemical Information and "
         "Modeling, 59(3), 947-961. https://doi.org/10.1021/acs.jcim.8b00712",
 "boyles": "Boyles, F., Deane, C. M., & Morris, G. M. (2020). Learning from the ligand: Using ligand-based "
           "features to improve binding affinity prediction. Bioinformatics, 36(3), 758-764. "
           "https://doi.org/10.1093/bioinformatics/btz665",
 "volkov": "Volkov, M., Turk, J.-A., Drizard, N., Martin, N., Hoffmann, B., Gaston-Mathe, Y., & Rognan, D. "
           "(2022). On the frustration to predict binding affinities from protein-ligand structures with deep "
           "neural networks. Journal of Medicinal Chemistry, 65(11), 7946-7958. "
           "https://doi.org/10.1021/acs.jmedchem.2c00487",
 "wu": "Wu, M.-H., Xie, Z., & Zhi, D. (2025). A Folding-Docking-Affinity framework for protein-ligand binding "
       "affinity prediction. Communications Chemistry, 8, 108. https://doi.org/10.1038/s42004-025-01506-1",
 "geirhos": "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, "
            "F. A. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11), "
            "665-673. https://doi.org/10.1038/s42256-020-00257-z",
 "cleverhans": "Lapuschkin, S., Waldchen, S., Binder, A., Montavon, G., Samek, W., & Muller, K.-R. (2019). "
               "Unmasking Clever Hans predictors and assessing what machines really learn. Nature "
               "Communications, 10(1), 1096. https://doi.org/10.1038/s41467-019-08987-4",
 "adebayo": "Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks "
            "for saliency maps. In Advances in Neural Information Processing Systems 31. arXiv:1810.03292. "
            "https://arxiv.org/abs/1810.03292",
 "damour": "D'Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., et al. (2020). "
           "Underspecification presents challenges for credibility in modern machine learning. "
           "arXiv:2011.03395. https://arxiv.org/abs/2011.03395",
 "xai": "Jimenez-Luna, J., Grisoni, F., & Schneider, G. (2020). Drug discovery with explainable artificial "
        "intelligence. Nature Machine Intelligence, 2(10), 573-584. "
        "https://doi.org/10.1038/s42256-020-00236-4",
 "rives": "Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., et al. (2021). Biological structure "
          "and function emerge from scaling unsupervised learning to 250 million protein sequences. "
          "Proceedings of the National Academy of Sciences, 118(15), e2016239118. "
          "https://doi.org/10.1073/pnas.2016239118",
 "esm2": "Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., "
         "Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). "
         "Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, "
         "379(6637), 1123-1130. https://doi.org/10.1126/science.ade2574",
 "af2": "Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., et al. (2021). Highly "
        "accurate protein structure prediction with AlphaFold. Nature, 596(7873), 583-589. "
        "https://doi.org/10.1038/s41586-021-03819-2",
 "attention": "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & "
              "Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing "
              "Systems 30. arXiv:1706.03762. https://arxiv.org/abs/1706.03762",
 "karplus": "Karplus, M., & McCammon, J. A. (2002). Molecular dynamics simulations of biomolecules. Nature "
            "Structural Biology, 9(9), 646-652. https://doi.org/10.1038/nsb0902-646",
 "amaro": "Amaro, R. E., Baudry, J., Chodera, J., Demir, O., McCammon, J. A., Miao, Y., & Smith, J. C. "
          "(2018). Ensemble docking in drug discovery. Biophysical Journal, 114(10), 2271-2278. "
          "https://doi.org/10.1016/j.bpj.2018.02.038",
 "vajda": "Vajda, S., Beglov, D., Wakefield, A. E., Egbert, M., & Whitty, A. (2018). Cryptic binding sites on "
          "proteins: Definition, detection, and druggability. Current Opinion in Chemical Biology, 44, 1-8. "
          "https://doi.org/10.1016/j.cbpa.2018.05.003",
 "pdbbind04": "Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). The PDBbind database: Collection of binding "
              "affinities for protein-ligand complexes with known three-dimensional structures. Journal of "
              "Medicinal Chemistry, 47(12), 2977-2980. https://doi.org/10.1021/jm030580l",
 "pdbbind05": "Wang, R., Fang, X., Lu, Y., Yang, C.-Y., & Wang, S. (2005). The PDBbind database: "
              "Methodologies and updates. Journal of Medicinal Chemistry, 48(12), 4111-4119. "
              "https://doi.org/10.1021/jm048957q",
 "lppdbbind": "Li, J., Guan, X., Zhang, O., Sun, K., Wang, Y., Bagni, D., & Head-Gordon, T. (2023). Leak Proof PDBBind: A reorganized dataset of protein-ligand complexes for more generalizable binding affinity prediction. arXiv:2308.09639 (version 3, 10 January 2026). https://arxiv.org/abs/2308.09639",
 "rcsb": "Burley, S. K., Bhikadiya, C., Bi, C., Bittrich, S., Chao, H., Chen, L., et al. (2023). RCSB Protein "
         "Data Bank (RCSB.org): Delivery of experimentally-determined PDB structures alongside one million "
         "computed structure models of proteins from artificial intelligence/machine learning. Nucleic Acids "
         "Research, 51(D1), D488-D508. https://doi.org/10.1093/nar/gkac1077",
 "nosek": "Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration "
          "revolution. Proceedings of the National Academy of Sciences, 115(11), 2600-2606. "
          "https://doi.org/10.1073/pnas.1708274114",
 "ecfp": "Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. Journal of Chemical Information "
         "and Modeling, 50(5), 742-754. https://doi.org/10.1021/ci100050t",
 "xgboost": "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the "
            "22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). "
            "https://doi.org/10.1145/2939672.2939785",
 "rf": "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. "
       "https://doi.org/10.1023/A:1010933404324",
 "betanll": "Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). On the pitfalls of heteroscedastic "
            "uncertainty estimation with probabilistic neural networks. In International Conference on "
            "Learning Representations. arXiv:2203.09168. https://arxiv.org/abs/2203.09168",
 "adamw": "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In International "
          "Conference on Learning Representations. arXiv:1711.05101. https://arxiv.org/abs/1711.05101",
 "ensembles": "Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive "
              "uncertainty estimation using deep ensembles. In Advances in Neural Information Processing "
              "Systems 30. arXiv:1612.01474. https://arxiv.org/abs/1612.01474",
 "conformal": "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free "
              "predictive inference for regression. Journal of the American Statistical Association, "
              "113(523), 1094-1111. https://doi.org/10.1080/01621459.2017.1307116",
 "hirschfeld": "Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R., & Coley, C. W. (2020). Uncertainty "
               "quantification using neural networks for molecular property prediction. Journal of Chemical "
               "Information and Modeling, 60(8), 3770-3780. https://doi.org/10.1021/acs.jcim.0c00502",
 "bootstrap": "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap (Monographs on "
              "Statistics and Applied Probability 57). Chapman & Hall.",
 "pytorch": "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: "
            "An imperative style, high-performance deep learning library. In Advances in Neural Information "
            "Processing Systems 32. arXiv:1912.01703. https://arxiv.org/abs/1912.01703",
 "rdkit": "Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org",
}
