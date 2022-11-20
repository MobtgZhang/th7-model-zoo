http = require("socket.http")
require("pl")
function download(result_dir)
    -- the file download file directory
    local result_dir = result_dir or "./"
    -- the file download website
    -- creat a table for the download
    local all_dataset = {}
    all_dataset["train_img_url"] = {
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",path.join(result_dir,"train-images-idx3-ubyte.gz")
    }
    all_dataset["train_label_url"]  = {
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",path.join(result_dir,"train-labels-idx1-ubyte.gz")
    }
    all_dataset["test_img_url"] = {
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",path.join(result_dir,"t10k-images-idx3-ubyte.gz")
    }
    all_dataset["test_label_url"]  = {
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",path.join(result_dir,"t10k-labels-idx1-ubyte.gz")
    }
    -- download the file
    -- print(all_dataset)
    for key_idx,pack_data in pairs(all_dataset) do
        local web_url = pack_data[1]
        local save_file = pack_data[2]
        local body,code = http.request(web_url)
        if not body then
            error(code)
        else
            print(string.format("Download the file %s",save_file))
            local fp = assert(io.open(save_file,"wb"))
            fp:write(body)
            fp:close()
            print(string.format("The file is saved in %s",save_file))
        end
    end
end

download()
