#!/usr/bin/python3
from enum import Enum
from os import path
from pwn import *
from struct import *
from optparse import OptionParser


__author__ = "Glenn McGuire (glen-mac)"
__contributor__ = "dhondta"
__source__ = "https://github.com/glen-mac/ELF-Packer"


parser = OptionParser()
parser.add_option("-f", "--file", help="the ELF file to pack")
parser.add_option("-v", "--verbose", default=False, action="store_true", help="verbose mode")
(options, args) = parser.parse_args()

ETYPE_DIC = {
    0: 'No file type',
    1: 'Relocatable file',
    2: 'Executable file',
    3: 'Shared object file',
    4: 'Core file'
}


# Enum of the section types
class SectionType(Enum):
    SHT_NULL = 0
    SHT_PROGBITS = 1
    SHT_SYMTAB = 2
    SHT_STRTAB = 3
    SHT_RELA = 4
    SHT_HASH = 5
    SHT_DYNAMIC = 6
    SHT_NOTE = 7
    SHT_NOBITS = 8
    SHT_REL = 9
    SHT_SHLIB = 10
    SHT_DYNSYM = 11
    SHT_LOPROC = 0x70000000
    SHT_HIPROC = 0x7fffffff
    SHT_LOUSER = 0x80000000
    SHT_HIUSER = 0xffffffff


class Section:
    def __init__(self, sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign,
                 sh_entsize):
        self.sh_name = sh_name
        self.sh_type = sh_type
        self.sh_flags = sh_flags
        self.sh_addr = sh_addr
        self.sh_offset = sh_offset
        self.sh_size = sh_size
        self.sh_link = sh_link
        self.sh_info = sh_info
        self.sh_addralign = sh_addralign
        self.sh_entsize = sh_entsize

    def __str__(self):
        return f"""[Start Section '{self.name}']
        sh_name      = {hex(self.sh_name)}
        sh_type      = {hex(self.sh_type)}
        sh_flags     = {hex(self.sh_flags)}
        sh_addr      = {hex(self.sh_addr)}
        sh_offset    = {hex(self.sh_offset)}
        sh_size      = {hex(self.sh_size)}
        sh_link      = {hex(self.sh_link)}
        sh_info      = {hex(self.sh_info)}
        sh_addralign = {hex(self.sh_addralign)}
        sh_entsize   = {hex(self.sh_entsize)}
        """


class Elf:
    def __init__(self, name="", data=[], bits='', size=0):
        self.name = name
        self.data = data
        self.bits = int(bits)
        self.EI_NIDENT = 16
        self.size = size
        self.ELF_EHDR_SZ = 36 + self.EI_NIDENT if self.bits == 32 else 48 + self.EI_NIDENT

    def parse_header(self):
        """
        Parse the ELF header of the binary

        e_ident:        marks the file as an object file
        e_type:         identifies the object file type
        e_machine:      specifies the required architecture of the file
        e_version:      identifies the object file version
        e_entry:        virtual address where the system first transfers control
        e_phoff:        program header table file offset in bytes
        e_shoff:        section header table offset in bytes
        e_flags:        holds processor specific flags associated with the file
        e_ehsize:       the elf header size in bytes
        e_phentsize:    size in bytes of one entry in program header table
        e_phnum:        number of entries in program header table
        e_shentsize:    size in bytes of one section header in section header table
        e_shnum:        number of entries in section header table
        e_shstrndx:     index of the section header table for string table
        """
        unpack_str = f"{self.EI_NIDENT}sHHIQQQIHHHHHH" if self.bits == 64 else f"{self.EI_NIDENT}sHHIIIIIHHHHHH"
        (self.e_ident, self.e_type, self.e_machine, self.e_version, self.e_entry, self.e_phoff, self.e_shoff,
         self.e_flags, self.e_ehsize, self.e_phentsize, self.e_phnum, self.e_shentsize, self.e_shnum,
         self.e_shstrndx) = unpack(unpack_str, self.data[:self.ELF_EHDR_SZ])
        logging.debug(f"entry point found:\t{hex(self.e_entry)}")
        logging.debug(f"object file type:\t{ETYPE_DIC[self.e_type]}")

    def parse_sections_header(self):
        """
        Parse sections of the Section Header Table

        sh_name:        index into the string table of the section name
        sh_type:        categorizes the sections contents and semantics
        sh_flags:       flags that describe miscellaneous attributes
        sh_addr:        virtual address of the first byte when in memory (if)
        sh_offset:      offset from start of file of first byte in the section
        sh_size:        the size of the section in bytes
        sh_link:        section header table index link
        sh_info:        holds extra information depending on section type
        sh_addralign:   dictates if the section has some form of size alignment
        sh_entsize:     size in bytes of each entry of section-fixed size table
        """
        unpack_str = "IIQQQQIIQQ" if self.bits == 64 else "IIIIIIIIII"
        # dictionary of arrays indexed by section type
        self.sections = {}
        section_header_sz = self.e_shnum * self.e_shentsize
        section_table = self.data[self.e_shoff:self.e_shoff+section_header_sz]
        # skip the first section in the section header table
        for sec_index in range(1, self.e_shnum):
            # unpack the section data
            (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize) = \
                unpack(unpack_str, section_table[sec_index*self.e_shentsize:(sec_index+1)*self.e_shentsize])
            # create the section
            sec = Section(sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign,
                          sh_entsize)
            if (not self.sections.get(sh_type)):
                self.sections[sh_type] = []
            self.sections[sh_type].append(sec)
            if sec_index == self.e_shstrndx:
                self.string_table_offset = sh_offset
        # add the section name to each section object
        for sec_type in self.sections.keys():
            for sec in self.sections.get(sec_type):
                sec.name = self.get_string(sec.sh_name)
                logging.debug(sec)

    def find_cave(self, required_size):
        # ensure that we don't look at 'SHT_NOBITS' sections
        for sec_type in self.sections.keys():
            for sec in self.sections.get(sec_type):
                if sec.sh_type == SectionType.SHT_NOBITS:
                    continue
                index = 0
                seen_nulls = 0
                checkpoint = 0
                try:
                    while (index < sec.sh_size):
                        char = self.data[sec.sh_offset + index]
                        index += 1
                        if char == 0:
                            seen_nulls += 1
                        else:
                            checkpoint = index
                            seen_nulls = 0
                        if seen_nulls == required_size:
                            break
                    if seen_nulls < required_size:
                        continue
                    logging.debug(f"""found a code cave in section: {sec.name} with
                            required size of {required_size} bytes at address
                            {hex(sec.sh_offset + checkpoint)} in the file. The address in memory
                            would be {hex(sec.sh_addr + index)}""")
                    return sec.sh_addr + checkpoint, sec.sh_offset + checkpoint
                except IndexError:  # occurs when being in an overlapping section whose section header size is out of
                    continue        #  the range of binary's size
        logging.debug("no code cave found")
        return None, None

    def get_string(self, index):
        elf_str = ''
        char = self.data[self.string_table_offset + index]
        while (char != 0):
            index += 1
            elf_str += chr(char)
            char = self.data[self.string_table_offset + index]
        return elf_str

    def get_section(self, name):
        for sec_type in self.sections.keys():
            for sec in self.sections.get(sec_type):
                if sec.name == name:
                    return sec

    def pack_code(self, key):
        text_sec = self.get_section('.text')
        for i in range(text_sec.sh_size):
            self.data[text_sec.sh_offset + i] ^= key

    def change_ep(self, new_ep):
        if self.bits == 32:
            self.data[24:24 + 4] = p32(new_ep)
        else:
            self.data[24:24 + 8] = p64(new_ep)

    def create_unpacker(self, xor_byte):
        text_sec = self.get_section('.text')
        if self.bits == 64:
            text_addr = text_sec.sh_addr & 0xFFFFFFFFFFFFF000
        else:
            text_addr = text_sec.sh_addr & 0xFFFFF000
        syscall_str = 'int 0x80' if self.bits == 32 else 'syscall'
        register_prefix = 'r' if self.bits == 64 else 'e'
        syscall_num = '0x7d' if self.bits == 32 else '10'
        syscall_reg_1 = 'di' if self.bits == 64 else 'bx'
        syscall_reg_2 = 'si' if self.bits == 64 else 'cx'
        syscall_reg_2 = 'si' if self.bits == 64 else 'cx'
        syscall_reg_2 = 'si' if self.bits == 64 else 'cx'
        integer_size = 4 if self.bits == 32 else 8
        file_offset = 1337
        offset_to_xor_byte = 22

        unpacker_asm = asm(f"""
        /* blah */
        call save_ip
        save_ip:

        /* open executable */
        mov rax, 2
        mov rdi, [{register_prefix}sp + {2 * integer_size}]
        push rdi # store the name
        mov rsi, O_RDONLY
        {syscall_str}
        mov r15, rax # store for closing

        /* mmap the executable */
        mov r8, rax
        mov rax, 9
        mov rdi, 0
        mov rsi, {self.size}
        mov rdx, PROT_READ
        mov r10, MAP_PRIVATE
        mov r9, 0
        {syscall_str}
        mov r14, rax

        /* open new executable */
        mov rax, 0x0062612f706d742f # /tmp/abc
        push rax

        /* remove old /tmp/ab */
        mov rax, 87
        mov rdi, rsp
        {syscall_str}

        /* continue open new executable */
        mov rax, 2
        mov rdi, rsp
        mov rsi, O_RDWR | O_CREAT
        mov rdx, 0x1FF
        {syscall_str}
        mov r12, rax

        mov rax, 77
        mov rdi, r12
        mov rsi, {self.size}
        {syscall_str}

        /* mmap the new executable */
        mov rax, 9
        mov rdi, 0
        mov rsi, {self.size}
        mov rdx, PROT_READ | PROT_WRITE
        mov r10, MAP_SHARED
        mov r8, r12
        mov r9, 0
        {syscall_str}
        mov r13, rax # store the mapping addr for encryption

        /* memcpy between mappings */
        mov rdi, rax         # the new file
        mov rsi, r14         # the executable file
        mov {register_prefix}cx, {self.size}  # the size of the file
        xor rax, rax
        copy:
            mov al, [rsi]
            mov [rdi], al
            inc rdi
            inc rsi
            dec rcx
            cmp rcx, 0
            jnz copy

        /* munmap the executable file */
        mov rax, 11
        mov rdi, r14
        mov rsi, {self.size}
        {syscall_str}

        /* close the executable */
        mov rax, 3
        mov rdi, r15
        {syscall_str}

        /* unlink the original executable file */
        mov rax, 87
        mov rdi, [rsp + 8]
        {syscall_str}

        /* move the new exec to where the old one was */
        mov rax, 82
        mov rdi, rsp
        mov rsi, [rsp + 8]
        {syscall_str}

        /* decrypt the new executable */
        mov rdi, r13    # the new file mapping addr
        add rdi, {text_sec.sh_offset} # increment to the text offset
        mov rsi, rdi          # the executable file
        mov {register_prefix}cx, {text_sec.sh_size}  # the size of the file
        xor rbx, rbx          # blank the register for the bytes
        mov bl, [rip + 0x9f]  # the xor byte for this current process
        cld
        recrypt_file1:
            lodsb
            xor rax, rbx
            stosb
            loop recrypt_file1

        /* replace the decrypt byte */
        mov rdi, r13
        mov rax, [rsp + 16]
        sub rax, 0x3ffe1f # minus 4, minus 0x400000 then plus 0x1ea for offset
        add rdi, rax
        /* get new random byte */
        mov rax, 318
        mov rsi, 1
        xor rdx, rdx
        /* end new random byte */
        syscall
        xor rbx, rbx
        mov rbx, [rdi]

        /* recrypt the new executable */
        mov rdi, r13    # the new file mapping addr
        add rdi, {text_sec.sh_offset} # increment to the text offset
        mov rsi, rdi          # the executable file
        mov {register_prefix}cx, {text_sec.sh_size}  # the size of the file
        cld
        recrypt_file2:
            lodsb
            xor rax, rbx
            stosb
            loop recrypt_file2

        /* munmap new exectuable */
        mov rax, 11
        mov rdi, r13
        mov rsi, {self.size}
        {syscall_str}

        /* close new executable */
        mov rax, 3
        mov rdi, r12
        {syscall_str}

        /* ~~~~ end polypacking ~~~~ */

        /* mprotect the current proc's text section */
        mov {register_prefix}ax, {syscall_num}
        mov {register_prefix}{syscall_reg_1}, {text_addr}
        mov {register_prefix}{syscall_reg_2}, {text_sec.sh_size}
        mov {register_prefix}dx, 0x7
        {syscall_str}

        mov {register_prefix}di, {text_sec.sh_addr}
        mov {register_prefix}si, {register_prefix}di
        mov {register_prefix}cx, {text_sec.sh_size}
        cld
        decrypt:
            lodsb
            xor al, {xor_byte}
            stosb
            loop decrypt

        mov {register_prefix}ax, {syscall_num}
        mov {register_prefix}{syscall_reg_1}, {text_addr}
        mov {register_prefix}{syscall_reg_2}, {text_sec.sh_size}
        mov {register_prefix}dx, 0x5
        {syscall_str}

        add rsp, 24
        xor rsi, rsi
        xor rdi, rdi
        xor rdx, rdx
        xor rcx, rcx
        xor rbx, rbx
        xor rax, rax
        push {self.e_entry}
        ret
        """)
        return unpacker_asm

    def write_unpacker(self, asm, off):
        self.data[unpacker_off:unpacker_off + len(asm)] = asm


if __name__ == '__main__':
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=[logging.INFO, logging.DEBUG][options.verbose])
    logging.getLogger("pwnlib").setLevel(logging.INFO)
    # load data into memory
    try:
        with open(options.file, 'rb') as f:
            elf_data = bytearray(f.read())
    except:
        print(f"ERROR: failed to open {options.file}")
        raise
        sys.exit(1)
    # check header
    if elf_data[:4] != b'\x7fELF':
        print(f"ERROR: {options.file} is not an ELF file")
        sys.exit(1)
    # check what arch we are packing for
    options.bits = ["32", "64"][elf_data[4] == 2]
    context.arch = ["i386", "amd64"][options.bits == "64"]
    # perform the crypting
    binary = Elf(name=options.file, data=elf_data, bits=options.bits, size=path.getsize(options.file))
    binary.parse_header()
    binary.parse_sections_header()
    xor_byte = ord(os.urandom(1))
    binary.pack_code(xor_byte)
    unpacker_asm = binary.create_unpacker(xor_byte)
    logging.debug(f"need {len(unpacker_asm)} bytes in a cave for unpacker")
    (unpacker_addr, unpacker_off) = binary.find_cave(len(unpacker_asm))
    if unpacker_addr:
        binary.change_ep(unpacker_addr)
        binary.write_unpacker(unpacker_asm, unpacker_off)
        # save packed binary to new file
        with open(f"{options.file}.packed", 'wb') as f:
            f.write(binary.data)
        # make new file executable
        st = os.stat(f"{options.file}.packed")
        os.chmod(f"{options.file}.packed", st.st_mode | stat.S_IEXEC)
    else:
        logging.error(f"cannot pack {options.file}")
