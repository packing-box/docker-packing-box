# Management

[*Packing Box*](https://github.com/dhondta/docker-packing-box) is designed to be used by researchers without the need to touch the code base. Therefore, it offers a tool, [`packing-box`](https://github.com/packing-box/docker-packing-box/blob/main/src/files/tools/packing-box), that allows to manage the most important parameters of the toolkit, including paths to YAML configurations for various assets like machine learning algorithms, packers, detectors, features and some others.

## Framework Configuraiton

The current configuration can be inspected using the `list` command as shown here below.

```console
$ packing-box list config 

Main

 • workspace     = /home/user/.packing-box                   
 • experiments   = /mnt/share/experiments                    
 • backup_copies = 3                                         
 • exec_timeout  = 10                                        

Definitions

 • algorithms  = /home/user/.packing-box/conf/algorithms.yml
 • alterations = /home/user/.packing-box/conf/alterations.yml
 • analyzers   = /home/user/.packing-box/conf/analyzers.yml  
 • detectors   = /home/user/.packing-box/conf/detectors.yml  
 • features    = /home/user/.packing-box/conf/features.yml   
 • packers     = /home/user/.packing-box/conf/packers.yml    
 • unpackers   = /home/user/.packing-box/conf/unpackers.yml  

Logging

 • lief_errors = False                                       
 • wine_errors = False                                       

Others

 • data           = /home/user/.packing-box/data             
 • hash_algorithm = sha256                                   

Parsers

 • default_parser = lief                                     
 • elf_parser     = lief                                     
 • macho_parser   = lief                                     
 • pe_parser      = lief                                     

```


