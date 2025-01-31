<div style="text-align: center;">

# MusicBoxDesigner

[![License](https://img.shields.io/github/license/BiologyHazard/MusicBoxDesigner?style=flat-square)](https://github.com/BiologyHazard/MusicBoxDesigner/blob/main/LICENSE)
[![Release](https://img.shields.io/github/release/BiologyHazard/MusicBoxDesigner?style=flat-square)](https://github.com/BiologyHazard/MusicBoxDesigner/releases/latest)

</div>

* [说明](#说明)
* [提示](#提示)
* [快速上手](#快速上手)
    * [安装](#安装)
        * [方法 1. 由可执行文件直接运行（仅 Windows 系统）](#方法-1-由可执行文件直接运行仅-windows-系统)
        * [方法 2. 从源码运行](#方法-2-从源码运行)
    * [教程](#教程)
* [代码示例](#代码示例)
* [Issue](#issue)
* [TODO](#todo)

# 说明

主要功能

- .mid, .emid, .fmp, .mcode 文件之间互相转换
- 生成纸带设计稿图片
- 统计音符数量和纸带长度

作者：[BioHazard](https://github.com/BiologyHazard)

- bilibili [Bio-Hazard](https://space.bilibili.com/37179776)
- QQ [3482991796](https://wpa.qq.com/msgrd?&uin=3482991796)
- QQ群 [586134350](https://qm.qq.com/cgi-bin/qm/qr?k=aM1lRdY9HvrQW3huC81hRmCQaE7CkyXh)

FairyMusicBox 官网：<http://www.fairymusicbox.com/>

FairyMusicBox 系列软件作者：[bilibili@调皮的码农](https://space.bilibili.com/40962795)

# 提示

FairyMusicBox 3.0.0 实现了本程序的几乎全部功能。如果您只是想进行一些基本的操作（生成纸带设计稿等），并且您没有超出 FairyMusicBox 许可协议的使用需求，建议使用 FairyMusicBox。

FairyMusicBox 3.0.0 生成的纸带设计稿是矢量图，可以无限放大而不失真。

如果 FairyMusicBox 的功能无法满足您的需求，或者您有超出 FairyMusicBox 许可协议的使用需求，那么，感谢您选择 MusicBoxDesigner，我们开始吧！

# 快速上手

入口程序 main.py 提供了

- convert（文件格式转换）
- draft（生成纸带设计稿图片）
- count（计算纸带长度和音符个数）

3种快捷操作，开箱即用！

## 安装

以下两种方式 <span style="font-size: 1.75em;">**任选其一**</span>

### 方法 1. 由可执行文件直接运行（仅 Windows 系统）

点击 [Latest Release](https://github.com/BiologyHazard/MusicBoxDesigner/releases/latest)，在页面下方的 Assets 中下载 Music-Box-Designer-\<version>.zip，并解压。

（但是不能直接双击运行，因为你还没告诉程序要干什么，接着往下看吧）

### 方法 2. 从源码运行

1. 获取源代码

    运行命令
    ```bash
    git clone https://github.com/BiologyHazard/MusicBoxDesigner.git
    ```

    如果报错或者超时，请点击 [Download ZIP](https://github.com/BiologyHazard/MusicBoxDesigner/archive/refs/heads/main.zip) 下载源代码并解压。

2. 安装 Python（3.12及以上）

    - Windows 系统

        按快捷键 Win + R，输入 cmd，回车，在弹出的窗口中输入
        ```bash
        python --version
        ```
        检查 Python 版本是否大于等于 3.12。如果报错或者版本过低，您需要按照下面的方法安装 Python。

        > 请自行前往 [Python 官网](https://www.python.org/) 下载 Python（3.12及以上）并将 Python 添加到环境变量（在安装过程中勾选 "Add Python to system PATH"）。
        >
        > 对大多数用户来说，您应该下载 Windows installer (64-bit)。

    - macOS 系统 以及 Linux 系统

        相信您有能力自己搞定！

    安装完成之后，运行命令

    ```bash
    python --version
    ```
    检查是否安装成功。

3. pip 安装依赖

    运行命令
    ```bash
    pip install -r requirements.txt
    ```

## 教程

无论你通过哪种方法完成了 [#安装](#安装) 中的操作，恭喜你！接下来只需要一行简单的命令就可以使用了。

如果您使用 Windows 系统，在入口程序所在的文件夹的**空白处**，按住 Shift，单击右键，在右键菜单中选择“在终端中打开(T)”，下面的命令请在这个终端中输入。

入口程序本身提供了丰富的帮助信息。运行下面的命令查看帮助。
```bash
python main.py --help
```

> 如果下载的是可执行文件例如 Music-Box-Designer-1.0.0.exe，则命令应当是
> ```bash
> .\Music-Box-Designer-1.0.0.exe --help
> ```
> 请将 "Music-Box-Designer-1.0.0.exe" 替换为实际的文件名，在接下来的命令中，都请自行把 `python main.py` 替换为 `.\Music-Box-Designer-1.0.0.exe`

- convert（转换文件格式）

    命令为
    ```bash
    python main.py convert source destination [-t TRANSPOSITION] [-o]
    ```
    支持 .emid, .fmp, .mcode 格式与 .mid 格式之间的互相转换。  
    不支持 .emid, .fmp, .mcode 格式之间的互相转换。
    - 参数 `source` 是源文件路径。可以使用 `directory/*.mid` 表示 directory 目录下的所有 midi 文件。
    - 参数 `destination` 是目标文件路径。可以仅指定格式（例如 `.fmp`）。
    - 可选参数 `-t TRANSPOSITION, --transposition TRANSPOSITION` 表示转调。
    - 可选参数 `-o, --overwrite` 表示允许覆盖现有的文件。

- draft（生成纸带设计稿图片）

    命令为
    ```bash
    python main.py draft file_path [settings_path] [-p] [-N NOTE_COUNT] [-t TRANSPOSITION] [-k] [-n] [-b BPM] [-T TITLE] [-S SUBTITLE] [-I MUSIC_INFO] [-B SHOW_BPM] [-s SCALE] [-o]
    ```
    - 参数 `file_path` 是源文件路径。支持 .mid, .emid, .fmp 格式。
    - 可选参数 `settings_path` 是纸带设计稿设置文件路径。默认为 `draft_settings.yml`。
    - 可选参数 `-p, --pdf` 表示输出为 PDF 文件，默认输出为 PNG 图片序列。
    - 可选参数 `-N NOTE_COUNT, --note-count NOTE_COUNT` 表示多少音八音盒，默认为30音（如果不指定这一参数，且 `file_path` 是 .fmp 文件，则会使用工程设置中的值）。
    - 可选参数 `-t TRANSPOSITION, --transposition TRANSPOSITION` 表示转调。
    - 可选参数 `-k, --keep-blank` 表示保留乐曲开头的空白。
    - 可选参数 `-n, --keep-near-notes` 表示保留过近而无法发声的音符。
    - 可选参数 `-b BPM, --bpm BPM` 指定乐曲速度。  
        设定此参数会使得 note 圆点的纵向间隔随着 midi 速度的变化而变化。  
        note 圆点间隔的缩放倍数 = `BPM` / midi的速度。  
        例如，midi 的速度被设定为 75bpm，`BPM` 设定为 100，则 note 圆点的间隔拉伸为 4/3 倍。  
        不提供此参数则忽略 midi 的速度信息，固定为 1拍 = 8毫米间隔。  
        当你以 `BPM` 的速度摇纸带的时候，相当于直接播放 midi。  
        该参数仅对 midi 生效，emid 和 fmp 会忽略此参数。
    - 可选参数 `-T TITLE, --title TITLE` 重写标题。
    - 可选参数 `-S SUBTITLE, --subtitle SUBTITLE` 重写副标题。
    - 可选参数 `-I MUSIC_INFO, --music-info MUSIC_INFO` 重写位于每栏右上角的乐曲信息。
    - 可选参数 `-B TEMPO_TEXT, --tempo-text TEMPO_TEXT` 指定乐曲速度信息的显示文本。
    - 可选参数 `-s SCALE, --scale SCALE` 指定 note 圆点纵向间距的缩放倍数。默认为 1（不缩放）。
    - 可选参数 `-o, --overwrite` 表示允许覆盖现有的文件。

- count（计算纸带长度和音符个数）

    命令为
    ```bash
    python main.py count file_path [-t TRANSPOSITION] [-k] [-n] [-b BPM] [-s SCALE]
    ```
    - 参数 `file_path` 是源文件路径。支持 .mid, .emid, .fmp 格式。
    - 其他可选参数的解释请参考 draft。

# 代码示例

以下代码可以在 music_box_designer/&#95;&#95;init&#95;&#95;.py 中找到。

- 把 examples/example.emid 转换成 midi 格式并保存到 examples/example.mid

    ```python
    from music_box_designer.emid import EmidFile

    emid_file = EmidFile.load_from_file('examples/example.emid')
    midi_file = emid_file.export_midi()
    midi_file.save('examples/example.mid')

    # or in a single line:
    # EmidFile.load_from_file('examples/example.emid').export_midi().save('examples/example.mid')
    ```

- 把 examples/example.midi 转换成 emid 格式并保存到 examples/example.emid

    ```python
    from mido import MidiFile
    from music_box_designer.emid import EmidFile

    midi_file = MidiFile('examples/example.mid')
    emid_file = EmidFile.from_midi(midi_file)
    emid_file.save_to_file('examples/example.emid')

    # or in a single line:
    # EmidFile.from_midi(MidiFile('examples/example.mid')).save_to_file('examples/example.emid')
    ```

- 对 fmp 的支持是类似的

    ```python
    from mido import MidiFile
    from music_box_designer.fmp import FmpFile

    FmpFile.load_from_file('examples/example.fmp').export_midi().save('examples/example.mid')
    # and
    FmpFile.new().import_midi(MidiFile('examples/example.mid')).save_to_file('examples/example.fmp')
    ```

- 同样支持 mcode

    ```python
    from mido import MidiFile
    from music_box_designer.mcode import MCodeFile

    MCodeFile.open('examples/example.mcode').export_midi().save('examples/example.mid')
    # and
    MCodeFile.from_midi(MidiFile('examples/example.mid')).save('examples/example.mcode')
    ```

- 从 examples/example.mid 生成纸带设计稿

    ```python
    import yaml
    from music_box_designer.draft import Draft, DraftSettings

    # load settings from draft_settings.yml
    with open('draft_settings.yml', 'rb') as fp:
        obj = yaml.safe_load(fp)
    settings: DraftSettings = DraftSettings.model_validate(obj)

    # or just create a DraftSettings instance with default or custom values
    # settings = DraftSettings(show_subtitle=False, show_bar_count=False)

    Draft.load_from_file('examples/example.mid').export_pics(
        settings=settings,
        title='Your title',
        subtitle='''Your subtitle''',
    ).save()
    ```

# Issue

如果在使用过程中遇到了问题，请点击页面顶部的 Issue → New Issue，并详细描述您的问题。

# TODO

- [x] 自定义水印
- [x] 显示小节号
- [x] 更多的快捷方法以及批处理方法
- [x] 用户友好的 README
- [ ] 自动打包
- [x] main.py
- [ ] draft 支持 fmp 标记
- [ ] interactive mode

祝使用愉快！
